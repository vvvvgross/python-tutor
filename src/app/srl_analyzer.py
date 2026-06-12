from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# Весовые коэффициенты для SPAS по фазам SRL
# (согласно формуле из главы 2.2 НИР)
# ============================================================
PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    # Для фазы planning важнее низкий HR, высокий latency (размышление), низкий TMR пока
    "planning": {
        "hr": -0.3,       # высокий hint_rate снижает вероятность планирования
        "tmr": -0.2,      # высокий TMR — уже не планирование
        "pi": 0.2,        # настойчивость небольшая на этапе планирования
        "latency": 0.3,   # долгое время отклика — думает, планирует
    },
    # Для фазы performance: активное решение задач
    "performance": {
        "hr": 0.1,        # подсказки допустимы
        "tmr": 0.4,       # прогресс в освоении
        "pi": 0.3,        # настойчивость
        "latency": 0.2,   # среднее время отклика
    },
    # Для фазы reflection: высокий TMR, низкая активность
    "reflection": {
        "hr": -0.2,       # меньше подсказок
        "tmr": 0.5,       # высокое мастерство
        "pi": 0.2,        # сохранение уровня
        "latency": 0.1,   # быстрее отвечает
    },
}


@dataclass
class TaskInteraction:
    """
    Базовая единица SRL‑анализа.

    Один TaskInteraction — это попытка решить конкретную задачу
    (или логически завершённый учебный шаг).
    """

    session_id: str
    task_id: str
    timestamp: float

    # SRL‑фаза на момент взаимодействия
    phase: str  # "planning" | "performance" | "reflection"

    # Результаты
    success: bool
    used_hint: bool
    attempts: int
    response_time: float  # в секундах
    error: bool = False  # была ли техническая/логическая ошибка


class SRLAnalyzer:
    """
    Анализатор саморегулируемого обучения (Self‑Regulated Learning).

    Вычисляет:
        - Task Mastery Rate (TMR) — доля успешно выполненных задач
        - Hint Rate (HR) — частота использования подсказок на задачу
        - Persistence Index (PI) — индекс настойчивости (попытки/время)
        - Engagement Level (EL) — уровень вовлечённости (интенсивность активности)

    Также определяет текущую SRL‑фазу: planning / performance / reflection.
    """

    def __init__(self) -> None:
        # Вся "история" по‑прежнему хранится в тьюторе / репозитории;
        # сам анализатор потоков данных не хранит.
        self.phase_override: Optional[str] = None  # устанавливается при смене шага

    # ------------------------------------------------------------------
    # Метрики по сессии
    # ------------------------------------------------------------------

    def compute_session_metrics(
        self, interactions: List[TaskInteraction]
    ) -> Dict[str, float]:
        """
        Вычисляет TMR / HR / PI / Engagement по списку взаимодействий.
        """
        if not interactions:
            return {
                "task_mastery_rate": 0.0,
                "hint_rate": 0.0,
                "persistence_index": 0.0,
                "engagement_level": 0.0,
            }

        total_tasks = len(interactions)
        successes = sum(1 for it in interactions if it.success)
        hints = sum(1 for it in interactions if it.used_hint)
        total_attempts = sum(max(1, it.attempts) for it in interactions)
        total_time = sum(max(0.01, it.response_time) for it in interactions)

        # Task Mastery Rate ~ доля задач, решённых успешно
        tmr = successes / max(1, total_tasks)

        # Hint Rate ~ число подсказок на задачу
        hr = hints / max(1, total_tasks)

        # Persistence Index: больше попыток и разумное время => выше настойчивость.
        # Нормируем грубо в [0;1].
        avg_attempts = total_attempts / max(1, total_tasks)
        avg_time = total_time / max(1, total_tasks)
        # Больше 4 попыток обычно уже сигнал о проблемах; ограничиваем.
        pi_raw = (avg_attempts / 4.0) * 0.6 + (min(avg_time, 60.0) / 60.0) * 0.4
        persistence_index = max(0.0, min(1.0, pi_raw))

        # Engagement Level: сколько взаимодействий приходится на минуту.
        ts_values = [it.timestamp for it in interactions]
        span = max(1.0, max(ts_values) - min(ts_values))
        interactions_per_minute = total_tasks / (span / 60.0)
        # Нормируем: 0..1, где 0 — <0.5 интеракций/мин, 1 — >= 5 интеракций/мин
        engagement_level = max(
            0.0, min(1.0, (interactions_per_minute - 0.5) / max(1e-3, 5.0 - 0.5))
        )

        return {
            "task_mastery_rate": float(tmr),
            "hint_rate": float(hr),
            "persistence_index": float(persistence_index),
            "engagement_level": float(engagement_level),
        }

    # ------------------------------------------------------------------
    # SRL‑фаза
    # ------------------------------------------------------------------

    def infer_phase(self, interactions: List[TaskInteraction]) -> str:
        """
        Оценивает текущую SRL‑фазу:
            - planning: начало сессии, мало задач, низкий TMR
            - performance: активное решение задач
            - reflection: высокая освоенность, снижение активности
        """
        if self.phase_override:
            return self.phase_override
        if not interactions:
            return "planning"

        metrics = self.compute_session_metrics(interactions)
        tmr = metrics["task_mastery_rate"]
        engagement = metrics["engagement_level"]
        num_interactions = len(interactions)
        
        # Планирование: начало сессии или низкий прогресс
        if num_interactions <= 2:
            return "planning"
        if tmr < 0.2 or (engagement < 0.15 and num_interactions < 5):
            return "planning"
        
        # Выполнение: активная работа; не «перескакиваем» сразу в рефлексию
        # Требуем минимум 4 взаимодействия и TMR < 0.8, чтобы оставаться в performance
        if tmr < 0.8 or num_interactions < 4:
            return "performance"
        
        # Рефлексия: высокий TMR и достаточно взаимодействий
        return "reflection"

    def get_srl_state(self, session_id: str, interactions: List[TaskInteraction]) -> Dict:
        """
        Высокоуровневый интерфейс, который описан в НИР:

            get_srl_state(session_id) -> {
                "session_id": ...,
                "phase": "planning" | "performance" | "reflection",
                "metrics": {TMR, HR, PI, Engagement}
            }
        """
        metrics = self.compute_session_metrics(interactions)
        phase = self.infer_phase(interactions)
        return {
            "session_id": session_id,
            "phase": phase,
            "metrics": metrics,
        }

    # ------------------------------------------------------------------
    # SPAS — SRL‑Phase Alignment Score (из главы 2.2 НИР)
    # ------------------------------------------------------------------

    def compute_behavioral_features(
        self, interactions: List[TaskInteraction]
    ) -> Dict[str, float]:
        """
        Вычисляет нормированные поведенческие метрики (b_t):
            - hr: hint_rate (0..1, ограничено)
            - tmr: task_mastery_rate (0..1)
            - pi: persistence_index (0..1)
            - latency: средняя нормированная задержка ответа (0..1)
        """
        metrics = self.compute_session_metrics(interactions)
        tmr = metrics["task_mastery_rate"]
        hr = min(1.0, metrics["hint_rate"])  # ограничим до 1
        pi = metrics["persistence_index"]

        # Латентность (среднее время ответа), нормируем в [0,1] (30 сек = 1)
        if interactions:
            avg_resp = sum(max(0.01, it.response_time) for it in interactions) / len(
                interactions
            )
            latency = min(1.0, avg_resp / 30.0)
        else:
            latency = 0.5  # нейтральное значение

        return {"hr": hr, "tmr": tmr, "pi": pi, "latency": latency}

    def compute_phase_probability(
        self, phase: str, features: Dict[str, float]
    ) -> float:
        """
        P(S_t | b_t) — вероятность фазы по поведенческим метрикам.
        Формула из НИР: softmax по скалярному произведению весов и признаков.
        """
        weights = PHASE_WEIGHTS.get(phase, {})
        score = sum(weights.get(k, 0.0) * features.get(k, 0.0) for k in features)
        return score  # вернём сырой score, softmax применим на всех фазах

    def compute_spas(
        self, chosen_phase: str, interactions: List[TaskInteraction]
    ) -> float:
        """
        SPAS (SRL‑Phase Alignment Score) — согласованность выбранной
        системой SRL‑фазы с фактическими поведенческими сигналами обучающегося.

        Возвращает P(chosen_phase | b_t) ∈ [0,1].
        """
        if not interactions:
            return 0.5  # нейтрально при отсутствии данных

        features = self.compute_behavioral_features(interactions)

        # Считаем score для каждой фазы
        scores = {}
        for ph in ["planning", "performance", "reflection"]:
            scores[ph] = self.compute_phase_probability(ph, features)

        # Softmax для нормализации в вероятности
        max_s = max(scores.values())
        exp_scores = {ph: math.exp(s - max_s) for ph, s in scores.items()}
        total = sum(exp_scores.values())
        probabilities = {ph: exp_scores[ph] / total for ph in exp_scores}

        return probabilities.get(chosen_phase, 0.0)


def interaction_from_dict(data: Dict) -> TaskInteraction:
    """
    Вспомогательная функция для восстановления TaskInteraction
    из словаря (например, загруженного из репозитория).
    """
    return TaskInteraction(
        session_id=data.get("session_id", ""),
        task_id=data.get("task_id", ""),
        timestamp=float(data.get("timestamp", time.time())),
        phase=data.get("phase", "planning"),
        success=bool(data.get("success", False)),
        used_hint=bool(data.get("used_hint", False)),
        attempts=int(data.get("attempts", 1)),
        response_time=float(data.get("response_time", 0.0)),
        error=bool(data.get("error", False)),
    )


def interaction_to_dict(interaction: TaskInteraction) -> Dict:
    """Обратное преобразование для удобства логирования в DataRepository."""
    return asdict(interaction)


