from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

from app.affective_monitor import AffectiveMonitor
from app.data_repository import get_data_repository
from app.llm_orchestrator import LLMOrchestrator
from app.moral_engine import MoralProfileEngine, MoralSchemaEngine
from app.srl_analyzer import SRLAnalyzer, TaskInteraction, interaction_to_dict


class DecisionEngine:
    """
    Подсистема принятия решений, описанная в НИР.

    Поток:
        1) Сбор состояния (SRL + эмоции + профиль + схемы)
        2) Генерация списка возможных действий
        3) Моральный фильтр
        4) Расчёт ΔK, ΔE, EC, U(A)
        5) Выбор A_t и генерация ответа через LLM Orchestrator
        6) Логирование в DataRepository
    """

    def __init__(self, moral_profile_engine: Optional[MoralProfileEngine] = None) -> None:
        self.srl_analyzer = SRLAnalyzer()
        self.affective_monitor = AffectiveMonitor()
        self.moral_schema_engine = MoralSchemaEngine()
        # Позволяем использовать общий инстанс профилей, чтобы UI видел изменения
        self.moral_profile_engine = moral_profile_engine or MoralProfileEngine()
        self.llm_orchestrator = LLMOrchestrator()
        self._repo = get_data_repository()

    # ------------------------------------------------------------------
    # Высокоуровневый API
    # ------------------------------------------------------------------

    def decide_and_generate(
        self,
        session_id: str,
        user_message: str,
        history: List[Dict],
        interactions: List[TaskInteraction],
        current_stage: int = 0,  # cur_moral_id: 0=мотивация, 1=рекурсия, 2=инкапсуляция, 3=финал
        goal_context: Optional[Dict] = None,  # данные из goal_sessions: goal, block, current_step, task_id
    ) -> Dict:
        """
        Основной метод:
            - анализирует текущее состояние
            - выбирает педагогическое действие
            - генерирует ответ через LLM Orchestrator

        Возвращает словарь:
            {
              "reply": str,
              "action": str,
              "srl_state": {...},
              "affective": {...},
              "profile": {...},
              "schema_id": str
            }
        """
        # 1) SRL‑состояние
        srl_state = self.srl_analyzer.get_srl_state(session_id, interactions)
        # Логируем SRL‑снимок
        self._repo.log_event(
            "srl",
            {
                "session_id": session_id,
                "phase": srl_state.get("phase"),
                "metrics": srl_state.get("metrics"),
            },
        )

        # 2) Аффективное состояние
        # Можно передать часть метрик как контекст (hint_rate и т.п.)
        affective = self.affective_monitor.analyze_emotions(
            user_message,
            context={
                "hint_rate": srl_state["metrics"].get("hint_rate", 0.0),
            },
        )
        # Логируем аффективное состояние отдельно
        self._repo.log_event(
            "affective",
            {
                "session_id": session_id,
                "state": affective,
            },
        )

        # 3) Моральный профиль и схема
        profile = self.moral_profile_engine.get_profile(session_id)
        active_schema = self.moral_schema_engine.get_active_schema(
            srl_state=srl_state,
            affective_state=affective,
            profile=profile,
        )

        # 4) Генерируем список возможных действий
        candidate_actions = self._generate_candidate_actions(srl_state, affective)

        # 5) Моральный фильтр
        filtered_actions = [
            a for a in candidate_actions if a in active_schema.allowed_actions
        ]
        if not filtered_actions:
            filtered_actions = ["support"]

        # 6) Оценка ΔK, ΔE, EC, U(A) для каждого действия
        utilities: List[Tuple[str, float, Dict[str, float]]] = []
        for action in filtered_actions:
            u, components = self._estimate_utility(
                action,
                srl_state,
                affective,
                profile,
                active_schema,
            )
            utilities.append((action, u, components))

        # Выбираем действие с максимальной полезностью
        utilities.sort(key=lambda x: x[1], reverse=True)
        best_action, best_u, best_components = utilities[0]

        # Shadow policies (без LLM-вызовов, только для аналитики)
        _neutral_srl = {
            "phase": "planning",
            "metrics": {"task_mastery_rate": 0.0, "hint_rate": 0.0, "progress_index": 0.0, "engagement_level": 0.5},
        }
        # Базовая схема «Навык» для аналитической shadow-policy без переключения eBICA
        _neutral_schema = next(
            (s for s in self.moral_schema_engine.schemas if s.id == "skill"),
            self.moral_schema_engine.schemas[-1],
        )
        # Baseline: нейтральный SRL + БЕЗ схемы + реальный affective
        # (тьютор без SRL и eBICA, но видит тот же студенческий контекст)
        _base_candidates = ["explain", "question", "hint", "support"]
        _base_utils = [(a, *self._estimate_utility(a, _neutral_srl, affective, profile, None)) for a in _base_candidates]
        _base_action, _base_u, _ = max(_base_utils, key=lambda x: x[1])
        # SRL-only: реальный SRL + реальный affective + нейтральная схема (нет eBICA)
        _srl_candidates = self._generate_candidate_actions(srl_state, affective)
        _srl_utils = [(a, *self._estimate_utility(a, srl_state, affective, profile, _neutral_schema)) for a in _srl_candidates]
        _srl_action, _srl_u, _ = max(_srl_utils, key=lambda x: x[1])

        # 7) Генерация ответа через LLM Orchestrator.
        #    Строим адаптивный контекст в стиле, совместимом с существующей историей.
        adaptive_context = self._build_adaptive_context(
            best_action, srl_state, affective, profile, current_stage, goal_context
        )
        messages = history[-6:] if len(history) > 6 else history
        messages = list(messages) + [
            {
                "role": "system",
                "content": adaptive_context,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        model_desc = self.llm_orchestrator.select_best_model(
            {"session_id": session_id, "emotional_safety_index": affective["emotional_safety_index"]}
        )
        llm_response = self.llm_orchestrator.generate(
            model_desc.id,
            {
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 260,
                "session_id": session_id,
                "emotional_safety_index": affective["emotional_safety_index"],
            },
        )

        reply_text = llm_response.get("text", "") or ""

        # 8) Диалог может фиксировать подсказку/рефлексию, но не успешное решение.
        # Результат задачи поступает только от серверной автопроверки.
        if best_action == "reflect":
            self.moral_profile_engine.update_profile(session_id, {"type": "reflection"})
        elif best_action == "hint":
            self.moral_profile_engine.update_profile(session_id, {"type": "hint", "used_hint": True})

        # 9) Логируем решение
        self._repo.log_event(
            "decision",
            {
                "session_id": session_id,
                "message_length": len(user_message or ""),
                "action": best_action,
                "utility": best_u,
                "utility_components": best_components,
                "srl_state": srl_state,
                "affective": affective,
                "profile": profile,
                "schema_id": active_schema.id,
                "shadow": {
                    "baseline": {"action": _base_action, "utility": round(_base_u, 4)},
                    "srl_only": {"action": _srl_action,  "utility": round(_srl_u, 4)},
                },
                "llm": {
                    "model_id": llm_response.get("model_id"),
                    "latency": llm_response.get("latency"),
                    "score": llm_response.get("score"),
                    "score_components": llm_response.get("score_components"),
                },
            },
        )

        return {
            "reply": reply_text,
            "action": best_action,
            "srl_state": srl_state,
            "affective": affective,
            "profile": profile,
            "schema_id": active_schema.id,
        }

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_candidate_actions(
        srl_state: Dict,
        affective: Dict,
    ) -> List[str]:
        """
        Формирует список потенциальных педагогических действий:
            - explain   (объяснение/теория)
            - question  (вопрос/проверка)
            - hint      (подсказка)
            - support   (мотивационная поддержка)
            - reflect   (приглашение к рефлексии)
            - simplify  (упрощение задания)
            - challenge (усложнение)
        """
        phase = srl_state.get("phase", "planning")
        fs = float(affective.get("frustration_score", 0.0))

        actions: List[str] = []
        if phase == "planning":
            actions.extend(["explain", "support", "question"])
        elif phase == "performance":
            actions.extend(["question", "hint", "support"])
        else:  # reflection
            actions.extend(["reflect", "support", "question"])

        # При высокой фрустрации добавляем упрощение
        if fs > 0.7:
            actions.append("simplify")
        # При низкой фрустрации и высоком TMR — можно усложнить
        if (
            srl_state["metrics"].get("task_mastery_rate", 0.0) > 0.8
            and fs < 0.4
        ):
            actions.append("challenge")

        # Убираем дубликаты
        return list(dict.fromkeys(actions))

    @staticmethod
    def _estimate_utility(
        action: str,
        srl_state: Dict,
        affective: Dict,
        profile: Dict,
        schema,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Грубая оценка полезности действия.

        Возвращает (U, {ΔK, ΔE, EC}).
        """
        tmr   = float(srl_state["metrics"].get("task_mastery_rate", 0.0))
        hr    = float(srl_state["metrics"].get("hint_rate", 0.0))
        pi    = float(srl_state["metrics"].get("progress_index", 0.0))
        eng   = float(srl_state["metrics"].get("engagement_level", 0.5))
        phase = srl_state.get("phase", "performance")
        fs    = float(affective.get("frustration_score", 0.0))

        # ΔK — ожидаемый прирост знания (зависит от SRL-метрик)
        if action == "explain":
            delta_k = 0.15 * (1.0 - 0.5 * tmr)        # менее полезно при высоком TMR
        elif action == "question":
            delta_k = 0.10 * (0.5 + eng)               # полезнее при высокой вовлечённости
        elif action == "hint":
            delta_k = 0.12 * (1.0 + 0.4 * hr)          # полезнее при высоком HR
        elif action == "support":
            delta_k = 0.05
        elif action == "simplify":
            delta_k = 0.08 * (1.0 - 0.3 * tmr)
        elif action == "challenge":
            delta_k = 0.20 * (0.3 + 0.7 * tmr)         # challenge ценен только при высоком TMR
        elif action == "reflect":
            delta_k = 0.10 * (1.0 + 0.5 * pi)
        else:
            delta_k = 0.05

        # ΔE — ожидаемое эмоциональное воздействие (зависит от FS)
        if action == "explain":
            delta_e = -0.05 * (1.0 + fs)
        elif action == "question":
            delta_e = 0.05 * fs                         # раздражает при высоком FS
        elif action == "hint":
            delta_e = -0.10 * (1.0 + fs)               # сильнее успокаивает при высоком FS
        elif action == "support":
            delta_e = -0.20 * (0.5 + fs)               # очень полезна при высоком FS
        elif action == "simplify":
            delta_e = -0.15 * (1.0 + fs)
        elif action == "challenge":
            delta_e = 0.05 + 0.15 * fs                 # вреден при высоком FS
        elif action == "reflect":
            delta_e = -0.05
        else:
            delta_e = 0.0

        # Фазовый бонус: SRL-фаза влияет на ценность действия
        phase_bonus = 0.0
        if phase == "planning" and action in ("explain", "question"):
            phase_bonus = 0.05
        elif phase == "performance" and action in ("hint", "challenge", "question"):
            phase_bonus = 0.05
        elif phase == "reflection" and action == "reflect":
            phase_bonus = 0.10

        # EC — этическая компонента: высокая если action соответствует схеме,
        # низкая если action вредит при высоком FS
        allowed = getattr(schema, "allowed_actions", []) if schema else []
        if allowed and action in allowed:
            ec = 0.9                                    # высокое при schema-соответствии
        elif fs > 0.7 and action in ("challenge", "question"):
            ec = 0.1                                    # критически низкое: вред при стрессе
        elif fs > 0.5 and action == "challenge":
            ec = 0.35
        else:
            ec = 0.7                                    # нейтральное

        # Бонус за schema alignment
        schema_bonus = 0.08 if (allowed and action in allowed) else 0.0

        # Адаптивные веса: при высоком FS эмоции важнее обучения (по модели Зиммерманна)
        if fs > 0.5:
            w_k  = max(0.2,  0.5 - 0.3 * fs)   # убывает: 0.35 при fs=0.5 → 0.2 при fs=1.0
            w_e  = min(0.6,  0.3 + 0.3 * fs)   # растёт:  0.45 при fs=0.5 → 0.6 при fs=1.0
            w_ec = 0.2
        else:
            w_k, w_e, w_ec = 0.5, 0.3, 0.2

        # Уменьшенный phase_bonus, чтобы emotional actions могли победить при высоком FS
        u = w_k * delta_k + w_e * (-delta_e) + w_ec * ec + phase_bonus * 0.4 + schema_bonus
        return float(u), {"delta_k": delta_k, "delta_e": delta_e, "ec": ec}

    @staticmethod
    def _build_adaptive_context(
        action: str,
        srl_state: Dict,
        affective: Dict,
        profile: Dict,
        current_stage: int = 0,
        goal_context: Optional[Dict] = None,
    ) -> str:
        """Формирует краткий адаптивный контекст для LLM.

        Если goal_context передан (новый flow через goal_sessions), строит
        динамический промпт на основе реальной цели и текущего шага студента.
        Иначе — безопасно просит восстановить тему сессии, не придумывая новую.
        """
        phase = srl_state.get("phase", "planning")
        tmr = srl_state["metrics"].get("task_mastery_rate", 0.0)
        hr = srl_state["metrics"].get("hint_rate", 0.0)
        fs = affective.get("frustration_score", 0.0)
        label = affective.get("label", "neutral")

        # ── Динамический контекст (новый flow) ──────────────────
        if goal_context:
            goal = goal_context.get("goal") or {}
            block = goal_context.get("block") or {}
            steps = block.get("steps") or []
            step_idx = goal_context.get("current_step", 0)
            task_id = goal_context.get("task_id")

            goal_title = goal.get("title", "Python")
            total_steps = len(steps)
            _level_labels = {"beginner": "новичок", "intermediate": "средний", "advanced": "продвинутый"}
            level_str = _level_labels.get(goal.get("level", "beginner"), "новичок")
            _pref_labels = {"practice": "практика", "theory": "теория", "balanced": "баланс"}
            pref_str = _pref_labels.get(goal.get("preference", "balanced"), "баланс")

            # Информация о текущем шаге
            if 0 <= step_idx < total_steps:
                step = steps[step_idx]
                step_title = step.get("title", "")
                step_type = step.get("type", "")
                step_content = step.get("content_md", "")
            else:
                step_title = ""
                step_type = ""
                step_content = ""

            topic_instruction = (
                f"Цель студента: {goal_title}. "
                f"Уровень: {level_str}. Фокус: {pref_str}. "
                f"Текущий шаг ({step_idx + 1}/{total_steps}): {step_title}"
            )
            if step_type:
                topic_instruction += f" (тип: {step_type})"
            topic_instruction += ". "
            if step_content:
                topic_instruction += f"{step_content} "
            if task_id:
                topic_instruction += f"Текущее задание: {task_id}. "

            _level_hint = {
                "beginner":     "Объясняй просто, без жаргона, давай больше примеров.",
                "intermediate": "Используй умеренную терминологию, можно ссылаться на базовые концепции.",
                "advanced":     "Будь лаконичен, используй профессиональную терминологию.",
            }.get(goal.get("level", "beginner"), "")
            _pref_hint = {
                "practice": "Делай акцент на написание кода, давай меньше теории.",
                "theory":   "Объясняй концепции подробно, приводи примеры из документации.",
                "balanced": "",
            }.get(goal.get("preference", "balanced"), "")

            style_hints = " ".join(filter(None, [_level_hint, _pref_hint]))

            return (
                f"Ты — адаптивный тьютор по Python. {topic_instruction}"
                f"SRL‑фаза: {phase}, TMR: {tmr:.2f}, HR: {hr:.2f}. "
                f"Эмоции: {label}, фрустрация: {fs:.2f}. "
                f"Действие: {action}. "
                f"{style_hints} "
                "Отвечай кратко (<100 слов), поддерживающе, в контексте цели и текущего шага студента."
            )

        # ── Безопасный fallback без подмены выбранной темы ───────────────
        return (
            "Ты — адаптивный тьютор по Python. Контекст выбранной темы временно недоступен. "
            f"SRL‑фаза: {phase}, TMR: {tmr:.2f}, HR: {hr:.2f}. "
            f"Эмоции: {label}, фрустрация: {fs:.2f}. Действие: {action}. "
            "Не придумывай новую тему и не предлагай несвязанное упражнение. "
            "Попроси пользователя уточнить текущую тему или восстановить учебную сессию. "
            "Отвечай кратко и поддерживающе."
        )

