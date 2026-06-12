import numpy as np
import logging
from pathlib import Path
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.base_moral_scheme import BaseMoralScheme
from app.oai_interface import Interface
from app.helper import (
    start_promt_dvt,
    first_space,
    from1to2,
    feelings1,
    second_space,
    from2to3,
    feelings2,
    third_space,
    from3to4,
    feelings3,
    fourth_space,
    feelings4,
)
from app.data_repository import get_data_repository
from app.decision_engine import DecisionEngine
from app.srl_analyzer import (
    SRLAnalyzer,
    TaskInteraction,
    interaction_to_dict,
)
from app.aem_metric import get_aem_calculator
import uuid

def create_logger(logger_name, log_dir, log_file, mode='a'):
    """Создает логгер с файловым обработчиком"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Удаляем старые обработчики, если они есть
    logger.handlers = []
    
    handler = logging.FileHandler(
        os.path.join(log_dir, log_file),
        mode=mode,
        encoding='utf-8'
    )
    
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def setup_session_loggers(session_id, log_base_dir="logs"):
    """
    Создает набор логгеров для сессии с отдельными файлами:
    - chat.log - сообщения пользователя
    - tutor_responses.log - ответы тьютора
    - metrics.log - метрики обучения
    - errors.log - ошибки и исключения
    - session.log - общая информация о сессии
    - terminal.log - команды терминала (если нужно)
    """
    log_dir = os.path.join(log_base_dir, session_id)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    loggers = {
        'chat': create_logger(f"chat_{session_id}", log_dir, "chat.log", mode='a'),
        'tutor': create_logger(f"tutor_{session_id}", log_dir, "tutor_responses.log", mode='a'),
        'metrics': create_logger(f"metrics_{session_id}", log_dir, "metrics.log", mode='a'),
        'errors': create_logger(f"errors_{session_id}", log_dir, "errors.log", mode='a'),
        'session': create_logger(f"session_{session_id}", log_dir, "session.log", mode='a'),
        'terminal': create_logger(f"terminal_{session_id}", log_dir, "terminal.log", mode='a'),
    }
    
    # Записываем начало сессии
    loggers['session'].info(f"=== Session started: {session_id} ===")
    loggers['session'].info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return loggers


class DummyVirtualTutor:
    def __init__(self, id):
        self.client_id = id
        self.interface = Interface()
        self.messages = [{"role": "system", "content": start_promt_dvt}]

    def generate_answer(self, replic):
        """Только генерация ответов, без обработки команд"""
        self.messages.append({"role": "user", "content": replic})
        
        try:
            response = self.interface.get_dummy_replic(self.messages)
            self.messages.append({"role": "assistant", "content": response})
            return response
        except Exception as e:
            return f"Error generating answer: {str(e)}"

class VirtualTutor:
    def __init__(self, id, moral_profile_engine=None):
        self.client_id = id
        self.interface = Interface()
        self.data_repo = get_data_repository()
        # Используем общий MoralProfileEngine, если он передан из SessionManager
        self.decision_engine = DecisionEngine(moral_profile_engine=moral_profile_engine)
        self.srl_analyzer = SRLAnalyzer()
        
        # Создание набора логгеров для сессии с отдельными файлами
        # Определяем базовую директорию для логов
        # В Docker контейнере: /app/logs, на хосте: ./logs (относительно корня проекта)
        if os.path.exists("/app/logs"):
            log_base_dir = "/app/logs"
        else:
            # На хосте: идем от src/app/ вверх на 2 уровня до корня проекта
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            log_base_dir = os.path.join(project_root, "logs")
        self.loggers = setup_session_loggers(self.client_id, log_base_dir)
        
        # Для обратной совместимости сохраняем старые имена
        self.logger_dialog = self.loggers['chat']  # Используем chat логгер для диалогов
        self.logger_metrics = self.loggers['metrics']
        
        # Инициализация моральных схем (4 этапа обучения)
        self.ms_list = [
            BaseMoralScheme(first_space, from1to2, feelings=feelings1),
            BaseMoralScheme(second_space, from2to3, feelings=feelings2),
            BaseMoralScheme(third_space, from3to4, feelings=feelings3),
            BaseMoralScheme(fourth_space, feelings=feelings4)
        ]
        
        # Состояние обучения
        self.last_replic = ""
        self.prev_moral_id = 0
        self.cur_moral_id = 0
        self.messages = [{"role": "assistant", "content": start_promt_dvt}]
        self.schemes = [False, False, False, False]  # Освоение моральных схем
        self.brain = [False, False, False, False]    # Готовность к переходу
        
        # Метрики прогресса (согласно курсовой работе)
        self.metrics = {
            'accuracy': 0.0,           # Доля успешно пройденных автотестов
            'hint_rate': 0.0,          # Среднее количество запросов подсказок
            'response_time': 0.0,      # Среднее время ответа
            'emotional_distance': 0.0, # Расстояние между appraisals и feelings
            'tests_passed': 0,         # Количество пройденных тестов
            'total_tests': 0,          # Общее количество тестов
            'hints_used': 0,           # Количество использованных подсказок
            'session_start_time': time.time(),
            'last_response_time': time.time()
        }
        
        # История взаимодействий для анализа
        self.interaction_history: List[Dict] = []

        # История учебных задач / шагов для SRL‑анализа
        self.task_interactions: List[TaskInteraction] = []

        # Последнее известное аффективное состояние (FS, ESI и т.п.)
        self.last_affective_state: Dict = {}
        self.interaction_history = []

        # Контекст цели/плана/шага из goal_sessions (инъектируется извне)
        self.goal_context: Optional[Dict] = None
        
        # Настройки адаптивного обучения
        self.adaptive_settings = {
            'difficulty_level': 1,     # Уровень сложности (1-5)
            'explanation_detail': 2,   # Детализация объяснений (1-3)
            'hint_frequency': 1,       # Частота подсказок (1-3)
            'practice_ratio': 0.8      # Соотношение практики к теории
        }

    def set_goal_context(self, ctx: dict):
        """Обновляет контекст цели/плана/шага из goal_sessions."""
        self.goal_context = ctx

    def update_metrics(self, interaction_type: str, success: bool = True,
                      response_time: float = None, hint_used: bool = False):
        """Обновление метрик прогресса"""
        current_time = time.time()
        
        # Сохраняем предыдущее значение emotional_distance для оценки прогресса
        prev_emotional_distance = self.metrics.get('emotional_distance', 1.0)
        
        if interaction_type == 'test':
            self.metrics['total_tests'] += 1
            if success:
                self.metrics['tests_passed'] += 1
            self.metrics['accuracy'] = self.metrics['tests_passed'] / self.metrics['total_tests']
            
        elif interaction_type == 'hint':
            if hint_used:
                self.metrics['hints_used'] += 1
            self.metrics['hint_rate'] = self.metrics['hints_used'] / max(1, self.metrics['total_tests'] + 1)
            
        elif interaction_type == 'response':
            if response_time:
                # Обновление среднего времени ответа
                if self.metrics['response_time'] == 0:
                    self.metrics['response_time'] = response_time
                else:
                    self.metrics['response_time'] = 0.9 * self.metrics['response_time'] + 0.1 * response_time
        
        # Обновление эмоционального расстояния
        if self.cur_moral_id < len(self.ms_list):
            appr_state = self.ms_list[self.cur_moral_id].get_appraisals_state()
            feel_state = self.ms_list[self.cur_moral_id].get_feelings_state()
            self.metrics['emotional_distance'] = self.ms_list[self.cur_moral_id].euc_dist(appr_state, feel_state)
        
        # Важное ограничение: учебная успешность изменяется только фактической
        # автопроверкой (interaction_type == 'test'). Диалог и эмоциональная
        # динамика логируются отдельно и не могут искусственно повышать TMR/accuracy.
        
        # Получаем все метрики для логирования
        session_duration = time.time() - self.metrics.get('session_start_time', time.time())
        
        # SRL метрики (TMR, PI, Engagement, фаза)
        srl_metrics = {}
        srl_phase = "planning"
        try:
            interactions = getattr(self, "task_interactions", []) or []
            srl_state = self.srl_analyzer.get_srl_state(self.client_id, interactions)
            srl_phase = srl_state.get("phase", "planning")
            srl_metrics = srl_state.get("metrics", {})
        except Exception as e:
            logging.warning(f"update_metrics: SRL calculation failed: {e}")
        
        # Аффективные метрики (FS, ESI)
        affective_state = getattr(self, "last_affective_state", {}) or {}
        frustration_score = affective_state.get("frustration_score", 0.0)
        emotional_safety_index = affective_state.get("emotional_safety_index", 0.0)
        
        # AEM метрика
        aem_score = 0.0
        try:
            from app.aem_metric import get_aem_calculator
            aem_calculator = get_aem_calculator()
            current_tmr = srl_metrics.get("task_mastery_rate", 0.0)
            initial_tmr = getattr(self, "initial_tmr", 0.0)
            spas = self.srl_analyzer.compute_spas(srl_phase, getattr(self, "task_interactions", []) or [])
            aem_result = aem_calculator.compute_session_aem(
                user_id=self.client_id,
                current_tmr=current_tmr,
                initial_tmr=initial_tmr,
                spas=spas,
            )
            aem_score = aem_result.get("aem", 0.0) if isinstance(aem_result, dict) else 0.0
        except Exception as e:
            logging.warning(f"update_metrics: AEM calculation failed: {e}")
        
        # Логирование всех метрик с дашборда
        self.loggers['metrics'].info(
            f"Metrics update - Type: {interaction_type}, Success: {success}, "
            f"Accuracy: {self.metrics['accuracy']:.3f}, "
            f"Hint Rate: {self.metrics['hint_rate']:.3f}, "
            f"Response Time: {self.metrics['response_time']:.3f}, "
            f"Emotional Distance: {self.metrics['emotional_distance']:.3f}, "
            f"SRL Phase: {srl_phase}, "
            f"TMR: {srl_metrics.get('task_mastery_rate', 0.0):.3f}, "
            f"PI: {srl_metrics.get('persistence_index', 0.0):.3f}, "
            f"Engagement: {srl_metrics.get('engagement_level', 0.0):.3f}, "
            f"FS: {frustration_score:.3f}, "
            f"ESI: {emotional_safety_index:.3f}, "
            f"AEM: {aem_score:.3f}, "
            f"Tests Passed: {self.metrics.get('tests_passed', 0)}, "
            f"Total Tests: {self.metrics.get('total_tests', 0)}, "
            f"Hints Used: {self.metrics.get('hints_used', 0)}, "
            f"Session Duration: {session_duration:.2f}s"
        )

    def _fallback_action_vector(self, intents_dict: Dict, phrase: str) -> np.ndarray:
        """
        Фоллбэк: если модель не вернула вектор интенций, генерируем небольшой
        детерминированный вектор, чтобы appraisals/feelings и emotional_distance
        могли обновляться.
        """
        size = len(intents_dict) if intents_dict else 4
        # Детерминированное зерно на основе текста
        seed = abs(hash(phrase)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(loc=0.0, scale=0.25, size=size)
        # Ограничим диапазон
        vec = np.clip(vec, -0.8, 0.8)
        # Избегаем полностью нулевого вектора
        if np.all(np.abs(vec) < 1e-3):
            vec[0] = 0.2
        return vec


    def adapt_learning_strategy(self):
        """Адаптация стратегии обучения на основе метрик"""
        # Адаптация уровня сложности
        if self.metrics['accuracy'] > 0.8 and self.metrics['emotional_distance'] < 0.25:
            self.adaptive_settings['difficulty_level'] = min(5, self.adaptive_settings['difficulty_level'] + 1)
        elif self.metrics['accuracy'] < 0.5 or self.metrics['emotional_distance'] > 0.5:
            self.adaptive_settings['difficulty_level'] = max(1, self.adaptive_settings['difficulty_level'] - 1)
        
        # Адаптация детализации объяснений
        if self.metrics['hint_rate'] > 1.5:
            self.adaptive_settings['explanation_detail'] = min(3, self.adaptive_settings['explanation_detail'] + 1)
        elif self.metrics['hint_rate'] < 0.5:
            self.adaptive_settings['explanation_detail'] = max(1, self.adaptive_settings['explanation_detail'] - 1)
        
        # Адаптация соотношения практики к теории
        if self.metrics['accuracy'] < 0.6:
            self.adaptive_settings['practice_ratio'] = min(0.9, self.adaptive_settings['practice_ratio'] + 0.1)
        else:
            self.adaptive_settings['practice_ratio'] = max(0.6, self.adaptive_settings['practice_ratio'] - 0.05)

    def should_progress_to_next_stage(self) -> bool:
        """Определение готовности к переходу на следующий этап"""
        interaction_count = self.metrics.get('total_tests', 0)
        
        # Специальная логика для перехода от этапа 0 (мотивация) к этапу 1 (рекурсия)
        if self.cur_moral_id == 0:
            # Переход к рекурсии: студент подтвердил готовность (проверил окружение)
            # Достаточно 1-2 взаимодействий, где студент выполнил проверку окружения
            return interaction_count >= 1
        
        # Для остальных переходов (рекурсия → инкапсуляция → финал)
        # Критерии перехода (смягчённые для более плавного прогресса)
        accuracy_threshold = 0.4  # Ещё более смягчено
        emotional_threshold = 0.7  # Ещё более смягчено
        response_time_threshold = 30.0  # секунд
        
        # Основные критерии
        basic_criteria = (
            self.metrics['accuracy'] >= accuracy_threshold and
            self.metrics['emotional_distance'] <= emotional_threshold and
            self.metrics['response_time'] <= response_time_threshold
        )
        
        # Альтернативные критерии: если взаимодействий достаточно и есть прогресс
        alternative_criteria = (
            interaction_count >= 3 and
            self.metrics['emotional_distance'] < 0.8
        )
        
        # Ещё один критерий: если моральная схема текущего этапа освоена
        schema_mastered = self.schemes[self.cur_moral_id] if self.cur_moral_id < len(self.schemes) else False
        
        return basic_criteria or alternative_criteria or schema_mastered

    def get_progress_summary(self) -> Dict:
        """Получение сводки прогресса студента"""
        return {
            'current_stage': self.cur_moral_id + 1,
            'total_stages': len(self.ms_list),
            'schemes_completed': sum(self.schemes),
            'brain_ready': sum(self.brain),
            'metrics': self.metrics.copy(),
            'adaptive_settings': self.adaptive_settings.copy(),
            'session_duration': time.time() - self.metrics['session_start_time']
        }

    def generate_answer(self, replic: str) -> str:
        """Совместимый вход: использует только актуальный DecisionEngine-поток."""
        return self.generate_answer_decision(replic)

    # ---------- НОВОЕ: интеграция DecisionEngine / SRL / аффективных метрик ----------

    def _register_task_interaction(self, success: bool, used_hint: bool, response_time: float) -> None:
        """
        Регистрирует учебное взаимодействие в SRL‑мониторе и файловом репозитории.
        """
        # Текущая SRL‑фаза до добавления новой записи
        current_state = self.srl_analyzer.get_srl_state(self.client_id, self.task_interactions)
        phase = current_state.get("phase", "planning")

        interaction = TaskInteraction(
            session_id=self.client_id,
            task_id=f"step-{len(self.task_interactions) + 1}",
            timestamp=time.time(),
            phase=phase,
            success=success,
            used_hint=used_hint,
            attempts=1,
            response_time=response_time,
            error=not success,
        )
        self.task_interactions.append(interaction)

        # Логируем событие задачи и актуальное SRL‑состояние
        try:
            self.data_repo.log_event("task", interaction_to_dict(interaction))
            self.data_repo.log_event(
                "srl",
                {
                    "session_id": self.client_id,
                    "phase": current_state.get("phase"),
                    "metrics": current_state.get("metrics"),
                },
            )
        except Exception:
            pass

    def generate_answer_decision(self, replic: str) -> str:
        """
        Альтернативный путь генерации ответа через DecisionEngine и LLMOrchestrator.

        Сохраняет 4‑этапную моральную схему (обновляет ms_list),
        добавляет SRL‑мониторинг и моральные профили.
        
        Также регистрирует действия в AEMCalculator для сбора обратной связи U_s.
        """
        start_time = time.time()
        self.last_replic = replic

        # Автоматическое обновление метрик по запросу подсказки
        try:
            text_lower = (replic or "").lower()
            if any(k in text_lower for k in ["подскажи", "подсказк", "hint", "дай совет", "help me"]):
                self.update_metrics("hint", True, hint_used=True)
        except Exception:
            pass

        # Обновление моральной схемы (как в generate_answer)
        intents = self.ms_list[self.cur_moral_id].get_base_intentions()
        action_vec = self.ms_list[self.cur_moral_id].oai_interface.get_composition(intents, replic)

        if action_vec is None or len(action_vec) == 0:
            action_vec = self._fallback_action_vector(intents, replic)

        # Обновляем схему ВСЕГДА (не только при fallback)
        self.ms_list[self.cur_moral_id].update_vectors(np.array(action_vec))
        appr_state = self.ms_list[self.cur_moral_id].get_appraisals_state()
        feel_state = self.ms_list[self.cur_moral_id].get_feelings_state()
        dist = self.ms_list[self.cur_moral_id].euc_dist(appr_state, feel_state)

        # Диалог не является доказательством успешного решения задачи.
        # Переходы и TMR/accuracy подтверждаются только автопроверкой кода.

        # Отмечаем освоение текущей схемы
        if dist < 0.25:
            self.schemes[self.cur_moral_id] = True

        # Адаптация стратегии обучения (как раньше)
        self.adapt_learning_strategy()

        # Получаем текущую SRL-фазу для AEM
        srl_state = self.srl_analyzer.get_srl_state(self.client_id, self.task_interactions)
        current_phase = srl_state.get("phase", "planning")
        current_tmr = srl_state.get("metrics", {}).get("task_mastery_rate", 0.0)

        # Вызов DecisionEngine + LLMOrchestrator
        try:
            decision_result = self.decision_engine.decide_and_generate(
                session_id=self.client_id,
                user_message=replic,
                history=self.messages,
                interactions=self.task_interactions,
                current_stage=self.cur_moral_id,  # Передаём текущий этап обучения
                goal_context=self.goal_context,    # Контекст цели/плана/шага
            )
            reply = decision_result.get("reply", "") or ""
            action_chosen = decision_result.get("action", "")
            # Обновляем кэш последнего аффективного состояния для API / UI
            self.last_affective_state = decision_result.get("affective", {}) or {}
        except Exception as e:
            # В случае исключения никогда не возвращаем пустой ответ в чат
            reply = (
                "Извини, сейчас у меня возникла внутренняя ошибка при обработке твоего сообщения. "
                "Попробуй, пожалуйста, повторить вопрос или переформулировать его."
            )
            # Логируем техническую деталь в файл ошибок, но не показываем её студенту
            try:
                self.loggers["errors"].error(f"DecisionEngine exception: {e}")
            except Exception:
                pass
            action_chosen = "support"

        # Страховка: даже если LLM вернул пустую строку без исключения,
        # никогда не отправляем в WebSocket пустой ответ — вместо этого даём понятный fallback.
        if not (reply and reply.strip()):
            reply = (
                "Похоже, сейчас сервис генерации ответов временно недоступен, "
                "и у меня не получилось сформировать осмысленный ответ. "
                "Попробуй, пожалуйста, ещё раз задать вопрос или немного переформулируй его."
            )
            try:
                self.loggers["errors"].error(
                    "LLM returned empty reply_text; fallback message was used."
                )
            except Exception:
                pass

        # Регистрируем действие в AEMCalculator для сбора обратной связи U_s
        # Только для действий hint и explain
        if action_chosen in ("hint", "explain"):
            try:
                aem_calculator = get_aem_calculator()
                action_id = f"action-{uuid.uuid4().hex[:8]}"
                aem_calculator.register_action(
                    user_id=self.client_id,
                    action_id=action_id,
                    action_type=action_chosen,
                    message=reply,
                    srl_phase=current_phase,
                    knowledge_before=current_tmr,
                )
                # Сохраняем ID последнего действия для UI
                self._last_action_id = action_id
                self._last_action_type = action_chosen
            except Exception:
                pass

        # Обновление истории и метрик
        self.messages.append({"role": "user", "content": replic})
        self.messages.append({"role": "assistant", "content": reply})

        response_time = time.time() - start_time
        self.update_metrics("response", True, response_time)
        
        # Обновляем emotional_distance после каждого ответа
        self.update_metrics("emotional", True)

        # Диалоговые сообщения не добавляются в TaskInteraction и не изменяют TMR.
        # Фактическое учебное взаимодействие регистрирует endpoint /api/autograde.

        # Обновляем high‑level историю для AI‑рекомендаций (последние 100 записей)
        self.interaction_history.append(
            {
                "timestamp": time.time(),
                "user": replic,
                "assistant": reply,
                "action": action_chosen,
            }
        )
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

        # Логирование диалога в отдельные файлы
        try:
            self.loggers['session'].info(f"Tutor turn completed; action={action_chosen}; user_chars={len(replic or '')}; reply_chars={len(reply or '')}")
            self.loggers['session'].info(f"Response time: {response_time:.3f}s")
        except Exception as e:
            self.loggers['errors'].error(f"Error logging dialog: {str(e)}")

        return reply

    
    
#---------- НОВОЕ
    def _recalculate_derived_metrics(self) -> None:
        """Перерасчет производных метрик из базовых счетчиков на всякий случай."""
        total_tests = self.metrics.get('total_tests', 0)
        tests_passed = self.metrics.get('tests_passed', 0)
        hints_used = self.metrics.get('hints_used', 0)

        # Перерасчет accuracy, если есть данные по тестам
        if total_tests > 0:
            self.metrics['accuracy'] = tests_passed / total_tests

        # Перерасчет hint_rate относительно количества задач (или 1, если задач ещё не было)
        self.metrics['hint_rate'] = hints_used / max(1, total_tests)

    def get_learning_recommendations(self, use_ai: bool = True) -> List[str]:
        """Получение расширенных рекомендаций для улучшения обучения на основе анализа метрик"""
        
        # Гарантируем консистентность производных метрик
        self._recalculate_derived_metrics()

        # Попытка получить AI-рекомендации
        if use_ai:
            try:
                ai_recommendations = self.interface.generate_ai_recommendations(
                    metrics=self.metrics,
                    adaptive_settings=self.adaptive_settings,
                    current_stage=self.cur_moral_id,
                    interaction_history=self.interaction_history
                )
                
                # Форматирование AI-рекомендаций
                formatted_recommendations = []
                
                # Добавляем анализ
                if ai_recommendations.get('analysis'):
                    formatted_recommendations.append(f"🤖 AI-анализ: {ai_recommendations['analysis']}")
                
                # Добавляем приоритетные рекомендации
                for rec in ai_recommendations.get('priority_recommendations', []):
                    formatted_recommendations.append(f"🚨 ПРИОРИТЕТ: {rec}")
                
                # Добавляем рекомендации по обучению
                for rec in ai_recommendations.get('learning_recommendations', []):
                    formatted_recommendations.append(f"💡 {rec}")
                
                # Добавляем мотивационный совет
                if ai_recommendations.get('motivational_advice'):
                    formatted_recommendations.append(f"🌟 {ai_recommendations['motivational_advice']}")
                
                # Добавляем следующие шаги
                for step in ai_recommendations.get('next_steps', []):
                    formatted_recommendations.append(f"➡️ {step}")
                
                # Добавляем информацию о времени улучшения
                if ai_recommendations.get('estimated_improvement_time'):
                    formatted_recommendations.append(f"⏱️ Ожидаемое время улучшения: {ai_recommendations['estimated_improvement_time']}")
                
                # AI-рекомендации не логируются в metrics.log (только обновления метрик)
                # Если AI вернул пусто — fallback на правила, чтобы UI не пустел
                if not formatted_recommendations:
                    return self._get_rule_based_recommendations()
                return formatted_recommendations[:8]  # Ограничиваем количество
                
            except Exception as e:
                self.loggers['errors'].error(f"AI recommendations failed: {str(e)}, falling back to rule-based")
                # Ошибки AI-рекомендаций не логируются в metrics.log
                # Fallback на rule-based рекомендации при ошибке AI
        
        # Rule-based рекомендации (fallback или при use_ai=False)
        return self._get_rule_based_recommendations()
    
    def _get_rule_based_recommendations(self) -> List[str]:
        """Получение рекомендаций на основе правил (fallback система)"""
        recommendations = []
        priority_recommendations = []
        
        # Анализ базовых метрик
        accuracy = self.metrics['accuracy']
        hint_rate = self.metrics['hint_rate']
        response_time = self.metrics['response_time']
        emotional_distance = self.metrics['emotional_distance']
        tests_passed = self.metrics['tests_passed']
        total_tests = self.metrics['total_tests']
        
        # Критические проблемы (высокий приоритет)
        if accuracy < 0.3 and hint_rate > 2.0:
            priority_recommendations.append("🚨 КРИТИЧНО: Очень низкая точность при частом использовании подсказок. Рекомендуется вернуться к основам и пройти базовые концепции заново")
        elif accuracy < 0.3:
            priority_recommendations.append("🚨 КРИТИЧНО: Критически низкая точность. Необходимо пересмотреть подход к обучению и обратиться за дополнительной помощью")
        
        # Анализ сочетаний метрик для более точных рекомендаций
        
        # 1. Анализ точности и времени ответа
        if accuracy < 0.5 and response_time > 20.0:
            recommendations.append("📊 Медленные ответы при низкой точности указывают на непонимание материала. Рекомендуется: 1) Повторить теорию, 2) Решать простые задачи для закрепления, 3) Не торопиться с ответами")
        elif accuracy > 0.8 and response_time < 5.0:
            recommendations.append("✅ Отличная точность и быстрые ответы! Вы готовы к более сложным задачам. Рекомендуется увеличить уровень сложности")
        elif accuracy > 0.7 and response_time > 15.0:
            recommendations.append("⏱️ Хорошая точность, но медленные ответы. Рекомендуется: 1) Практиковаться в быстром решении знакомых задач, 2) Развивать автоматизм в базовых операциях")
        
        # 2. Анализ использования подсказок и эмоционального состояния
        if hint_rate > 1.5 and emotional_distance > 0.4:
            recommendations.append("😰 Частое использование подсказок при высоком эмоциональном напряжении. Рекомендуется: 1) Делать перерывы между задачами, 2) Начинать с более простых задач для восстановления уверенности, 3) Практиковать техники релаксации")
        elif hint_rate < 0.5 and emotional_distance > 0.6:
            recommendations.append("💪 Редкое использование подсказок при высоком напряжении может указывать на перфекционизм. Рекомендуется: 1) Не бояться просить помощи, 2) Помнить, что ошибки - часть обучения")
        elif hint_rate > 2.0 and emotional_distance < 0.2:
            recommendations.append("🤔 Частое использование подсказок при спокойном состоянии. Рекомендуется: 1) Больше времени тратить на самостоятельное решение, 2) Анализировать, почему нужны подсказки")
        
        # 3. Анализ прогресса обучения
        if total_tests > 10:
            recent_accuracy = accuracy  # В реальной системе здесь был бы расчет по последним тестам
            if recent_accuracy > accuracy + 0.1:
                recommendations.append("📈 Положительная динамика! Ваши результаты улучшаются. Продолжайте в том же направлении")
            elif recent_accuracy < accuracy - 0.1:
                recommendations.append("📉 Отрицательная динамика. Рекомендуется: 1) Пересмотреть стратегию обучения, 2) Обратиться к преподавателю, 3) Сделать перерыв")
        
        # 4. Анализ готовности к переходу на следующий этап
        if self.should_progress_to_next_stage():
            recommendations.append("🎯 Отличные результаты! Вы готовы к переходу на следующий этап обучения")
        else:
            missing_criteria = []
            if accuracy < 0.8:
                missing_criteria.append(f"точность ({accuracy:.2f} < 0.8)")
            if emotional_distance > 0.25:
                missing_criteria.append(f"эмоциональная стабильность ({emotional_distance:.2f} > 0.25)")
            if response_time > 15.0:
                missing_criteria.append(f"скорость ответа ({response_time:.1f}s > 15s)")
            
            if missing_criteria:
                recommendations.append(f"🎯 Для перехода на следующий этап нужно улучшить: {', '.join(missing_criteria)}")
        
        # 5. Персонализированные рекомендации на основе текущего этапа
        stage_recommendations = self._get_stage_specific_recommendations()
        recommendations.extend(stage_recommendations)
        
        # 6. Рекомендации по адаптивным настройкам
        adaptive_recommendations = self._get_adaptive_recommendations()
        recommendations.extend(adaptive_recommendations)
        
        # 7. Рекомендации по времени обучения
        session_duration = time.time() - self.metrics['session_start_time']
        if session_duration > 3600:  # Более часа
            recommendations.append("⏰ Длительная сессия обучения. Рекомендуется сделать перерыв для лучшего усвоения материала")
        elif session_duration < 300:  # Менее 5 минут
            recommendations.append("⚡ Короткая сессия. Для эффективного обучения рекомендуется заниматься не менее 15-20 минут")
        
        # 8. Рекомендации по балансу практики и теории
        if self.adaptive_settings['practice_ratio'] > 0.9:
            recommendations.append("📚 Слишком много практики. Рекомендуется больше времени уделить изучению теории")
        elif self.adaptive_settings['practice_ratio'] < 0.6:
            recommendations.append("💻 Слишком много теории. Рекомендуется больше практических упражнений")
        
        # Объединение рекомендаций с приоритетами
        all_recommendations = priority_recommendations + recommendations
        
        # Ограничение количества рекомендаций для избежания перегрузки
        return all_recommendations[:8] if len(all_recommendations) > 8 else all_recommendations
    
    def _get_stage_specific_recommendations(self) -> List[str]:
        """Получение рекомендаций, специфичных для текущего этапа обучения"""
        recommendations = []
        current_stage = self.cur_moral_id
        
        if current_stage == 0:  # Первый этап
            if self.metrics['accuracy'] < 0.6:
                recommendations.append("🌱 На начальном этапе важно заложить прочный фундамент. Рекомендуется: 1) Медленно и внимательно изучать базовые концепции, 2) Много практиковаться с простыми примерами")
        elif current_stage == 1:  # Второй этап
            if self.metrics['hint_rate'] > 1.0:
                recommendations.append("🔧 На втором этапе важно развивать самостоятельность. Рекомендуется: 1) Больше времени тратить на самостоятельное решение, 2) Анализировать свои ошибки")
        elif current_stage == 2:  # Третий этап
            if self.metrics['response_time'] > 12.0:
                recommendations.append("⚡ На третьем этапе важно развивать скорость. Рекомендуется: 1) Практиковаться в быстром решении знакомых задач, 2) Развивать автоматизм")
        elif current_stage == 3:  # Четвертый этап
            if self.metrics['emotional_distance'] > 0.3:
                recommendations.append("🎯 На финальном этапе важно сохранять уверенность. Рекомендуется: 1) Вспомнить свои успехи, 2) Подготовиться к применению знаний на практике")
        
        return recommendations
    
    def _get_adaptive_recommendations(self) -> List[str]:
        """Получение рекомендаций по адаптивным настройкам"""
        recommendations = []
        
        # Рекомендации по уровню сложности
        if self.adaptive_settings['difficulty_level'] == 1 and self.metrics['accuracy'] > 0.7:
            recommendations.append("📈 Уровень сложности можно увеличить. Вы справляетесь с текущими задачами")
        elif self.adaptive_settings['difficulty_level'] == 5 and self.metrics['accuracy'] < 0.6:
            recommendations.append("📉 Уровень сложности слишком высок. Рекомендуется снизить для лучшего понимания")
        
        # Рекомендации по детализации объяснений
        if self.adaptive_settings['explanation_detail'] == 1 and self.metrics['hint_rate'] > 1.0:
            recommendations.append("📝 Рекомендуется увеличить детализацию объяснений для лучшего понимания")
        elif self.adaptive_settings['explanation_detail'] == 3 and self.metrics['hint_rate'] < 0.3:
            recommendations.append("⚡ Объяснения слишком детальные. Можно упростить для ускорения обучения")
        
        return recommendations
    
    def get_ai_learning_analysis(self) -> dict:
        """Получение детального AI-анализа обучения с расширенной информацией"""
        try:
            # Гарантируем актуальность производных метрик
            self._recalculate_derived_metrics()

            ai_recommendations = self.interface.generate_ai_recommendations(
                metrics=self.metrics,
                adaptive_settings=self.adaptive_settings,
                current_stage=self.cur_moral_id,
                interaction_history=self.interaction_history
            )
            
            # Добавляем дополнительную информацию
            analysis_data = {
                'ai_recommendations': ai_recommendations,
                'current_metrics': self.metrics.copy(),
                'adaptive_settings': self.adaptive_settings.copy(),
                'current_stage': self.cur_moral_id + 1,
                'total_stages': len(self.ms_list),
                'session_duration': time.time() - self.metrics['session_start_time'],
                'ready_for_next_stage': self.should_progress_to_next_stage(),
                'timestamp': time.time()
            }
            
            # AI-анализ не логируется в metrics.log (только обновления метрик)
            
            return analysis_data
            
        except Exception as e:
            self.loggers['errors'].error(f"Detailed AI analysis failed: {str(e)}")
            # Ошибки AI-анализа не логируются в metrics.log
            return {
                'error': str(e),
                'fallback_used': True,
                'current_metrics': self.metrics.copy(),
                'adaptive_settings': self.adaptive_settings.copy(),
                'timestamp': time.time()
            }
    
    def get_learning_insights(self) -> dict:
        """Получение инсайтов и трендов обучения"""
        # Гарантируем актуальность производных метрик
        self._recalculate_derived_metrics()

        insights = {
            'performance_trend': self._analyze_performance_trend(),
            'learning_patterns': self._analyze_learning_patterns(),
            'strengths_weaknesses': self._identify_strengths_weaknesses(),
            'recommended_focus_areas': self._get_focus_areas(),
            'motivation_level': self._assess_motivation_level()
        }
        
        return insights
    
    def _analyze_performance_trend(self) -> str:
        """Анализ тренда производительности"""
        if self.metrics['total_tests'] < 3:
            return "Недостаточно данных для анализа тренда"
        
        # Простая логика для демонстрации
        if self.metrics['accuracy'] > 0.7:
            return "Положительный тренд - точность выше среднего"
        elif self.metrics['accuracy'] < 0.4:
            return "Отрицательный тренд - требуется дополнительная поддержка"
        else:
            return "Стабильный тренд - постепенное улучшение"
    
    def _analyze_learning_patterns(self) -> dict:
        """Анализ паттернов обучения"""
        return {
            'hint_dependency': 'Высокая' if self.metrics['hint_rate'] > 1.5 else 'Низкая',
            'response_speed': 'Быстрая' if self.metrics['response_time'] < 10 else 'Медленная',
            'emotional_stability': 'Стабильная' if self.metrics['emotional_distance'] < 0.3 else 'Нестабильная',
            'practice_focus': 'Практика' if self.adaptive_settings['practice_ratio'] > 0.7 else 'Теория'
        }
    
    def _identify_strengths_weaknesses(self) -> dict:
        """Выявление сильных и слабых сторон"""
        strengths = []
        weaknesses = []
        
        if self.metrics['accuracy'] > 0.8:
            strengths.append("Высокая точность выполнения")
        elif self.metrics['accuracy'] < 0.5:
            weaknesses.append("Низкая точность выполнения")
        
        if self.metrics['response_time'] < 10:
            strengths.append("Быстрые ответы")
        elif self.metrics['response_time'] > 20:
            weaknesses.append("Медленные ответы")
        
        if self.metrics['hint_rate'] < 0.5:
            strengths.append("Самостоятельность")
        elif self.metrics['hint_rate'] > 2.0:
            weaknesses.append("Зависимость от подсказок")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def _get_focus_areas(self) -> List[str]:
        """Определение областей для фокуса"""
        focus_areas = []
        
        if self.metrics['accuracy'] < 0.6:
            focus_areas.append("Улучшение понимания базовых концепций")
        
        if self.metrics['hint_rate'] > 1.5:
            focus_areas.append("Развитие самостоятельности в решении задач")
        
        if self.metrics['response_time'] > 15:
            focus_areas.append("Ускорение процесса решения")
        
        if self.metrics['emotional_distance'] > 0.4:
            focus_areas.append("Работа с эмоциональным состоянием")
        
        return focus_areas if focus_areas else ["Продолжение текущего подхода"]
    
    def _assess_motivation_level(self) -> str:
        """Оценка уровня мотивации"""
        # Простая эвристика на основе метрик
        if self.metrics['accuracy'] > 0.7 and self.metrics['emotional_distance'] < 0.3:
            return "Высокая"
        elif self.metrics['accuracy'] < 0.4 or self.metrics['emotional_distance'] > 0.6:
            return "Низкая"
        else:
            return "Средняя"