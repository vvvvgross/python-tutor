"""
AEM — Adaptive Educational Merit

Интегральная метрика эффективности адаптивного действия тьютора,
объединяющая три компоненты (согласно главе 2.2 НИР):

    AEM = α₁·ΔK + α₂·U_s + α₃·SPAS

где:
    - ΔK   — прирост знаний (разность вероятности усвоения до и после)
    - U_s  — субъективная полезность ответа тьютора (обратная связь от пользователя)
    - SPAS — SRL-Phase Alignment Score (корректность выбора SRL-фазы)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import defaultdict


# Весовые коэффициенты (α₁, α₂, α₃) — настраиваются эмпирически
DEFAULT_WEIGHTS = {
    "alpha_k": 0.4,   # вес прироста знаний
    "alpha_us": 0.35, # вес субъективной полезности
    "alpha_spas": 0.25,  # вес согласованности SRL-фазы
}


@dataclass
class TutorAction:
    """
    Запись о действии тьютора для отслеживания обратной связи.
    """
    action_id: str
    user_id: str
    timestamp: float
    action_type: str  # "hint" | "explain" | "question" | "support" | etc.
    message: str
    srl_phase: str  # фаза SRL на момент действия
    
    # Обратная связь (заполняется позже)
    feedback_received: bool = False
    feedback_useful: Optional[bool] = None  # True = полезно, False = нет
    feedback_timestamp: Optional[float] = None
    
    # Для расчёта ΔK
    knowledge_before: float = 0.0  # P(знания до)
    knowledge_after: Optional[float] = None  # P(знания после)


@dataclass
class FeedbackRecord:
    """
    Запись обратной связи от пользователя.
    """
    action_id: str
    user_id: str
    useful: bool  # True = "Да", False = "Нет"
    timestamp: float = field(default_factory=time.time)


class AEMCalculator:
    """
    Калькулятор метрики AEM (Adaptive Educational Merit).
    
    Отвечает за:
    - Отслеживание действий тьютора (hint/explain)
    - Сбор обратной связи U_s
    - Расчёт ΔK (прирост знаний)
    - Интеграцию с SPAS
    - Вычисление итогового AEM
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        
        # Хранилище действий по user_id -> list[TutorAction]
        self._actions: Dict[str, List[TutorAction]] = defaultdict(list)
        
        # Последнее действие, ожидающее обратной связи (для быстрого доступа)
        self._pending_feedback: Dict[str, TutorAction] = {}
        
        # История U_s по сессиям
        self._us_history: Dict[str, List[float]] = defaultdict(list)
        
        # История AEM по сессиям
        self._aem_history: Dict[str, List[Dict]] = defaultdict(list)
    
    # ------------------------------------------------------------------
    # Регистрация действий тьютора
    # ------------------------------------------------------------------
    
    def register_action(
        self,
        user_id: str,
        action_id: str,
        action_type: str,
        message: str,
        srl_phase: str,
        knowledge_before: float = 0.0,
    ) -> TutorAction:
        """
        Регистрирует действие тьютора.
        Если action_type — "hint" или "explain", помечает его как ожидающее обратной связи.
        """
        action = TutorAction(
            action_id=action_id,
            user_id=user_id,
            timestamp=time.time(),
            action_type=action_type,
            message=message,
            srl_phase=srl_phase,
            knowledge_before=knowledge_before,
        )
        self._actions[user_id].append(action)
        
        # Действия hint и explain требуют обратной связи
        if action_type in ("hint", "explain"):
            self._pending_feedback[user_id] = action
        
        return action
    
    def has_pending_feedback(self, user_id: str) -> bool:
        """Проверяет, есть ли действие, ожидающее обратной связи."""
        return user_id in self._pending_feedback
    
    def get_pending_action(self, user_id: str) -> Optional[TutorAction]:
        """Возвращает действие, ожидающее обратной связи."""
        return self._pending_feedback.get(user_id)
    
    # ------------------------------------------------------------------
    # Обработка обратной связи
    # ------------------------------------------------------------------
    
    def record_feedback(
        self,
        user_id: str,
        useful: bool,
        action_id: Optional[str] = None,
    ) -> Optional[FeedbackRecord]:
        """
        Записывает обратную связь от пользователя.
        
        Args:
            user_id: ID пользователя
            useful: True если полезно, False если нет
            action_id: опционально, ID конкретного действия
        
        Returns:
            FeedbackRecord или None если нет ожидающего действия
        """
        # Находим действие для обратной связи
        if action_id:
            action = next(
                (a for a in self._actions[user_id] if a.action_id == action_id),
                None
            )
        else:
            action = self._pending_feedback.get(user_id)
        
        if not action:
            return None
        
        # Обновляем действие
        action.feedback_received = True
        action.feedback_useful = useful
        action.feedback_timestamp = time.time()
        
        # Удаляем из pending
        if user_id in self._pending_feedback and self._pending_feedback[user_id] == action:
            del self._pending_feedback[user_id]
        
        # Рассчитываем U_s (1.0 если полезно, 0.0 если нет)
        u_s = 1.0 if useful else 0.0
        self._us_history[user_id].append(u_s)
        
        feedback = FeedbackRecord(
            action_id=action.action_id,
            user_id=user_id,
            useful=useful,
        )
        
        return feedback
    
    # ------------------------------------------------------------------
    # Расчёт компонент AEM
    # ------------------------------------------------------------------
    
    def compute_delta_k(
        self,
        knowledge_before: float,
        knowledge_after: float,
    ) -> float:
        """
        ΔK — прирост знаний.
        Формула: ΔK = P(знания после) - P(знания до)
        Нормируется в [0, 1].
        """
        delta = knowledge_after - knowledge_before
        # Нормируем: если прирост отрицательный, считаем 0
        return max(0.0, min(1.0, delta))
    
    def compute_us_session(self, user_id: str) -> float:
        """
        U_s — средняя субъективная полезность за сессию.
        Формула: U_s = u / u_max, где u_max = 1.0
        """
        history = self._us_history.get(user_id, [])
        if not history:
            return 0.5  # нейтральное значение при отсутствии данных
        return sum(history) / len(history)
    
    def compute_us_recent(self, user_id: str, n: int = 5) -> float:
        """
        U_s — субъективная полезность за последние n действий.
        """
        history = self._us_history.get(user_id, [])
        if not history:
            return 0.5
        recent = history[-n:] if len(history) >= n else history
        return sum(recent) / len(recent)
    
    # ------------------------------------------------------------------
    # Расчёт AEM
    # ------------------------------------------------------------------
    
    def compute_aem(
        self,
        delta_k: float,
        u_s: float,
        spas: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Вычисляет интегральную метрику AEM.
        
        Формула: AEM = α₁·ΔK + α₂·U_s + α₃·SPAS
        """
        w = weights or self.weights
        alpha_k = w.get("alpha_k", 0.4)
        alpha_us = w.get("alpha_us", 0.35)
        alpha_spas = w.get("alpha_spas", 0.25)
        
        aem = alpha_k * delta_k + alpha_us * u_s + alpha_spas * spas
        return max(0.0, min(1.0, aem))
    
    def compute_session_aem(
        self,
        user_id: str,
        current_tmr: float,
        initial_tmr: float,
        spas: float,
    ) -> Dict:
        """
        Вычисляет AEM для сессии.
        
        Args:
            user_id: ID пользователя
            current_tmr: текущий Task Mastery Rate
            initial_tmr: начальный TMR сессии
            spas: SRL-Phase Alignment Score
        
        Returns:
            Dict с компонентами и итоговым AEM
        """
        # ΔK на основе прироста TMR
        delta_k = self.compute_delta_k(initial_tmr, current_tmr)
        
        # U_s из истории обратной связи
        u_s = self.compute_us_session(user_id)
        
        # Итоговый AEM
        aem = self.compute_aem(delta_k, u_s, spas)
        
        result = {
            "delta_k": round(delta_k, 4),
            "u_s": round(u_s, 4),
            "spas": round(spas, 4),
            "aem": round(aem, 4),
            "weights": self.weights.copy(),
        }
        
        # Сохраняем в историю
        self._aem_history[user_id].append({
            "timestamp": time.time(),
            **result,
        })
        
        return result
    
    # ------------------------------------------------------------------
    # Получение статистики
    # ------------------------------------------------------------------
    
    def get_feedback_stats(self, user_id: str) -> Dict:
        """
        Возвращает статистику обратной связи для пользователя.
        """
        actions = self._actions.get(user_id, [])
        feedbackable = [a for a in actions if a.action_type in ("hint", "explain")]
        with_feedback = [a for a in feedbackable if a.feedback_received]
        positive = [a for a in with_feedback if a.feedback_useful]
        
        total = len(feedbackable)
        received = len(with_feedback)
        positive_count = len(positive)
        
        return {
            "total_feedbackable_actions": total,
            "feedback_received": received,
            "positive_feedback": positive_count,
            "negative_feedback": received - positive_count,
            "feedback_rate": received / max(1, total),
            "positive_rate": positive_count / max(1, received) if received > 0 else 0.0,
            "average_us": self.compute_us_session(user_id),
        }
    
    def get_aem_history(self, user_id: str) -> List[Dict]:
        """Возвращает историю AEM для пользователя."""
        return self._aem_history.get(user_id, [])
    
    def get_latest_aem(self, user_id: str) -> Optional[Dict]:
        """Возвращает последний расчёт AEM."""
        history = self._aem_history.get(user_id, [])
        return history[-1] if history else None


# Singleton-экземпляр для глобального использования
_aem_calculator: Optional[AEMCalculator] = None


def get_aem_calculator() -> AEMCalculator:
    """Возвращает глобальный экземпляр AEMCalculator."""
    global _aem_calculator
    if _aem_calculator is None:
        _aem_calculator = AEMCalculator()
    return _aem_calculator

