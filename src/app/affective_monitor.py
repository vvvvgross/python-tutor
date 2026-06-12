from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

from app.oai_interface import Interface


@dataclass
class AffectiveState:
    """Сводка аффективного состояния для одного сообщения / шага."""

    frustration_score: float  # FS, 0..1
    emotional_safety_index: float  # ESI, 0..1 (чем выше, тем безопаснее поднимать нагрузку)
    label: str  # 'positive' | 'neutral' | 'frustrated' | 'bored' | 'confused' | ...


class AffectiveMonitor:
    """
    Аффективный мониторинг:
        - вычисляет Frustration Score (FS)
        - вычисляет Emotional Safety Index (ESI)
        - распознаёт тон сообщения при помощи LLM / эвристик.

    Интерфейс, описанный в НИР:

        analyze_emotions(message) -> {FS, ESI, label}
    """

    def __init__(self, interface: Optional[Interface] = None) -> None:
        # Можно переиспользовать уже созданный Interface из VirtualTutor
        self.interface = interface or Interface()

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def analyze_emotions(
        self,
        message: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Основной метод мониторинга.

        :param message: текст последнего сообщения студента
        :param context: дополнительные признаки (ошибки, задержки, hint_rate и т.п.)
        """
        context = context or {}

        # 1) Быстрая эвристика по тексту (ключевые слова)
        base_label = self._heuristic_label(message)

        # 2) LLM‑классификация (если доступна) — более точная, но не критичная
        llm_label = self._llm_classify_label(message) or base_label

        # 3) Frustration Score (FS) — на основе:
        #    - текста
        #    - серии ошибок / подсказок (hint_rate, error_streak и т.п.)
        fs = self._compute_frustration_score(llm_label, context)

        # 4) Emotional Safety Index (ESI) — обратная метрика "напряжения"
        esi = max(0.0, min(1.0, 1.0 - fs))

        return {
            "frustration_score": float(fs),
            "emotional_safety_index": float(esi),
            "label": llm_label,
        }

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_label(message: str) -> str:
        """Грубая текстовая эвристика, не требующая LLM."""
        text = (message or "").lower()
        if any(w in text for w in ["не понимаю", "бесит", "надоело", "это сложно", "я тупой"]):
            return "frustrated"
        if any(w in text for w in ["скучно", "нудно", "неинтересно"]):
            return "bored"
        if any(w in text for w in ["класс", "отлично", "понятно", "спасибо"]):
            return "positive"
        return "neutral"

    def _llm_classify_label(self, message: str) -> Optional[str]:
        """
        Обращается к LLM через существующий Interface для классификации тона.
        При любых сетевых/LLM‑ошибках возвращает None, чтобы не ломать основной поток.
        """
        try:
            prompt = f"""
Определи эмоциональный тон следующего сообщения студента.
Сообщение: \"{message}\"

Верни ОДНО слово на английском, строго из списка:
    positive, neutral, frustrated, bored, confused.
"""
            body = {
                "model": getattr(self.interface, "model_name", "deepseek-chat"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 4,
            }
            result = self.interface._make_api_request("/chat/completions", body)  # type: ignore[attr-defined]
            if not result:
                return None
            content = (
                result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )
            for label in ["positive", "neutral", "frustrated", "bored", "confused"]:
                if label in content:
                    return label
            return None
        except Exception:
            return None

    @staticmethod
    def _compute_frustration_score(label: str, context: Dict) -> float:
        """
        FS ∈ [0;1]. Учитываем:
            - LLM‑label (qualitative)
            - error_streak / hint_rate / long_pauses из контекста
        """
        base_map = {
            "positive": 0.05,
            "neutral": 0.2,
            "bored": 0.5,
            "confused": 0.7,
            "frustrated": 0.85,
        }
        fs = base_map.get(label, 0.2)

        # Усиливаем фрустрацию при накоплении ошибок / подсказок
        error_streak = float(context.get("error_streak", 0.0))
        hint_rate = float(context.get("hint_rate", 0.0))
        long_pause = bool(context.get("long_pause", False))

        if error_streak >= 3:
            fs += 0.1
        if hint_rate > 1.5:
            fs += 0.1
        if long_pause:
            fs += 0.05

        return max(0.0, min(1.0, fs))


