from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.data_repository import get_data_repository


@dataclass
class MoralSchema:
    """
    Моральная схема — набор правил, определяющих допустимые реакции тьютора
    в определённом состоянии SRL/эмоций.
    """

    id: str
    name: str
    description: str

    # Пороговые значения по метрикам
    max_frustration: float  # FS верхняя граница
    min_safety: float  # ESI нижняя граница

    # Допустимые педагогические действия
    allowed_actions: List[str]

    # Вес важности этического соответствия (0..1)
    ethical_weight: float = 1.0


class MoralSchemaEngine:
    """
    Менеджер моральных схем.

    В простейшем варианте — фиксированный набор профилей,
    "зашитый" в конфиг (кодом), как описано в НИР.
    """

    def __init__(self) -> None:
        self.schemas: List[MoralSchema] = self._default_schemas()

    @staticmethod
    def _default_schemas() -> List[MoralSchema]:
        """Четыре схемы, используемые в модели ВКР: мотивация, внимание, навык, мастерство."""
        return [
            MoralSchema(
                id="motivation", name="Мотивация",
                description="Признаки фрустрации или повторяющихся ошибок: сохранить готовность продолжать работу.",
                max_frustration=1.0, min_safety=0.0,
                allowed_actions=["support", "simplify", "hint", "question"], ethical_weight=1.0,
            ),
            MoralSchema(
                id="attention", name="Внимание",
                description="Потеря фокуса или неясный запрос: вернуть обучающегося к текущей цели.",
                max_frustration=0.75, min_safety=0.25,
                allowed_actions=["question", "simplify", "support", "reflect"], ethical_weight=0.9,
            ),
            MoralSchema(
                id="skill", name="Навык",
                description="Затруднение относится к конкретной конструкции программы: дать адресную помощь.",
                max_frustration=0.65, min_safety=0.35,
                allowed_actions=["explain", "hint", "question", "support"], ethical_weight=0.85,
            ),
            MoralSchema(
                id="mastery", name="Мастерство",
                description="Высокая успешность и автономность: поддержать самостоятельность и развитие.",
                max_frustration=0.35, min_safety=0.65,
                allowed_actions=["challenge", "reflect", "question", "explain"], ethical_weight=0.8,
            ),
        ]


    def get_active_schema(
        self,
        srl_state: Dict,
        affective_state: Dict,
        profile: Dict,
    ) -> MoralSchema:
        """Выбор схемы на основе SRL-фазы, учебных метрик и аффективного состояния."""
        fs = float(affective_state.get("frustration_score", 0.0))
        phase = str(srl_state.get("phase", "planning"))
        metrics = srl_state.get("metrics", {}) or {}
        tmr = float(metrics.get("task_mastery_rate", 0.0))
        hr = float(metrics.get("hint_rate", 0.0))
        pi = float(metrics.get("persistence_index", 0.0))

        by_id = {schema.id: schema for schema in self.schemas}
        if fs >= 0.65 or (tmr < 0.35 and hr >= 0.5):
            return by_id["motivation"]
        if phase == "reflection" and tmr >= 0.75 and hr <= 0.35 and fs <= 0.35:
            return by_id["mastery"]
        if phase == "planning" or (pi < 0.2 and tmr < 0.6):
            return by_id["attention"]
        return by_id["skill"]



class MoralProfileEngine:
    """
    Управление моральными профилями студентов.

    Профиль описывается четырьмя параметрами в [0;1]:
        - responsibility
        - integrity
        - autonomy
        - reflection

    См. НИР, раздел о таксономии моральных профилей.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, Dict[str, float]] = {}
        self._repo = get_data_repository()

    # ------------------------------------------------------------------
    # Работа с профилем
    # ------------------------------------------------------------------

    @staticmethod
    def _default_profile() -> Dict[str, float]:
        return {
            "responsibility": 0.5,
            "integrity": 0.5,
            "autonomy": 0.5,
            "reflection": 0.5,
        }

    def get_profile(self, student_id: str) -> Dict[str, float]:
        if student_id not in self._profiles:
            self._profiles[student_id] = self._default_profile()
        return self._profiles[student_id]

    def update_profile(self, student_id: str, event: Dict) -> Dict[str, float]:
        """
        Обновляет профиль студента на основе события.

        Ожидаемые ключи события (по НИР — это лишь одна из возможных реализаций):
            - type: 'task_result' | 'hint' | 'cheating' | ...
            - success: bool
            - used_hint: bool
        """
        profile = self.get_profile(student_id)
        etype = event.get("type", "")
        success = bool(event.get("success", False))
        used_hint = bool(event.get("used_hint", False))

        # Простейшие эвристики:
        if etype == "task_result":
            if success:
                profile["responsibility"] = min(1.0, profile["responsibility"] + 0.02)
                profile["autonomy"] = min(1.0, profile["autonomy"] + (0.01 if not used_hint else 0.0))
            else:
                profile["responsibility"] = max(0.0, profile["responsibility"] - 0.01)

        if etype == "hint":
            if used_hint:
                profile["autonomy"] = max(0.0, profile["autonomy"] - 0.01)

        if etype == "reflection":
            profile["reflection"] = min(1.0, profile["reflection"] + 0.02)

        if etype == "cheating":
            profile["integrity"] = max(0.0, profile["integrity"] - 0.1)

        # Логируем обновление профиля
        self._repo.log_event(
            "profile",
            {
                "student_id": student_id,
                "event": event,
                "profile": profile.copy(),
            },
        )

        return profile

    def classify_profile(self, student_id: str) -> str:
        """
        Классифицирует профиль в один из укрупнённых типов
        (описанных в НИР). Эвристически.
        """
        p = self.get_profile(student_id)
        r, i, a, refl = (
            p["responsibility"],
            p["integrity"],
            p["autonomy"],
            p["reflection"],
        )

        if r > 0.7 and a > 0.7 and i > 0.6:
            return "autonomous"
        if r > 0.6 and i > 0.6 and a < 0.4:
            return "formal"
        if i < 0.4:
            return "risky"
        if r < 0.4 and a < 0.4:
            return "disoriented"
        return "balanced"


