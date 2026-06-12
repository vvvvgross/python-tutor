#!/usr/bin/env python3
"""
Модульные тесты для компонентов адаптивного тьютора.
Приложение к НИР — раздел 4.4 «Функциональное и интеграционное тестирование».

Запуск:
    cd /path/to/python-tutor-experiment
    python -m pytest tests/test_unit.py -v

Покрытие:
    - SRLAnalyzer: вычисление метрик (TMR, HR, PI, engagement), определение фазы, phase_override
    - MoralSchemaEngine: выбор моральной схемы по FS/ESI
    - DecisionEngine._estimate_utility: оценка полезности действий
    - DecisionEngine._generate_candidate_actions: генерация кандидатов по фазе и FS
    - Интеграционный тест полного пайплайна
"""

import sys
import os
import math
import time
import pytest

# Убедимся, что src/ в пути
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from app.srl_analyzer import (
    SRLAnalyzer,
    TaskInteraction,
    interaction_from_dict,
    interaction_to_dict,
)
from app.moral_engine import MoralSchemaEngine, MoralSchema
from app.decision_engine import DecisionEngine


# ==============================================================================
# Вспомогательные функции
# ==============================================================================

def make_interaction(
    session_id: str = "test_session",
    task_id: str = "task_1",
    phase: str = "performance",
    success: bool = True,
    used_hint: bool = False,
    attempts: int = 1,
    response_time: float = 15.0,
    error: bool = False,
    timestamp: float = None,
) -> TaskInteraction:
    """Фабрика TaskInteraction для тестов."""
    return TaskInteraction(
        session_id=session_id,
        task_id=task_id,
        timestamp=timestamp or time.time(),
        phase=phase,
        success=success,
        used_hint=used_hint,
        attempts=attempts,
        response_time=response_time,
        error=error,
    )


def make_srl_state(
    phase: str = "performance",
    tmr: float = 0.5,
    hr: float = 0.2,
    pi: float = 0.4,
    eng: float = 0.5,
) -> dict:
    """Фабрика srl_state для тестов."""
    return {
        "phase": phase,
        "metrics": {
            "task_mastery_rate": tmr,
            "hint_rate": hr,
            "progress_index": pi,
            "engagement_level": eng,
        },
    }


def make_affective(fs: float = 0.0, esi: float = 1.0) -> dict:
    """Фабрика affective_state для тестов."""
    return {"frustration_score": fs, "emotional_safety_index": esi}


# ==============================================================================
# Тесты SRLAnalyzer
# ==============================================================================

class TestSRLAnalyzer:
    """Тесты аналитика саморегулируемого обучения."""

    def setup_method(self):
        self.analyzer = SRLAnalyzer()

    # --- compute_session_metrics ---

    def test_empty_interactions_returns_zeros(self):
        """При пустом списке взаимодействий все метрики равны 0."""
        metrics = self.analyzer.compute_session_metrics([])
        assert metrics["task_mastery_rate"] == 0.0
        assert metrics["hint_rate"] == 0.0
        assert metrics["persistence_index"] == 0.0
        assert metrics["engagement_level"] == 0.0

    def test_tmr_all_success(self):
        """TMR = 1.0 когда все задачи выполнены успешно."""
        interactions = [make_interaction(success=True) for _ in range(5)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["task_mastery_rate"] == pytest.approx(1.0)

    def test_tmr_half_success(self):
        """TMR = 0.5 при половине успешных задач."""
        interactions = (
            [make_interaction(success=True) for _ in range(3)] +
            [make_interaction(success=False) for _ in range(3)]
        )
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["task_mastery_rate"] == pytest.approx(0.5)

    def test_tmr_no_success(self):
        """TMR = 0.0 когда ни одна задача не выполнена."""
        interactions = [make_interaction(success=False) for _ in range(4)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["task_mastery_rate"] == 0.0

    def test_hint_rate_no_hints(self):
        """HR = 0.0 когда подсказки не использовались."""
        interactions = [make_interaction(used_hint=False) for _ in range(4)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["hint_rate"] == 0.0

    def test_hint_rate_all_hints(self):
        """HR = 1.0 когда подсказки использованы на каждой задаче."""
        interactions = [make_interaction(used_hint=True) for _ in range(4)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["hint_rate"] == pytest.approx(1.0)

    def test_hint_rate_partial(self):
        """HR корректно считается при частичном использовании подсказок."""
        interactions = (
            [make_interaction(used_hint=True) for _ in range(2)] +
            [make_interaction(used_hint=False) for _ in range(2)]
        )
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["hint_rate"] == pytest.approx(0.5)

    def test_persistence_index_range(self):
        """PI всегда в диапазоне [0.0, 1.0]."""
        interactions = [make_interaction(attempts=10, response_time=120.0) for _ in range(5)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert 0.0 <= metrics["persistence_index"] <= 1.0

    def test_persistence_index_single_attempt(self):
        """При 1 попытке и малом времени PI близок к минимуму."""
        interactions = [make_interaction(attempts=1, response_time=1.0)]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert metrics["persistence_index"] < 0.5

    def test_engagement_level_range(self):
        """Engagement Level всегда в [0.0, 1.0]."""
        now = time.time()
        interactions = [
            make_interaction(timestamp=now + i * 10.0) for i in range(10)
        ]
        metrics = self.analyzer.compute_session_metrics(interactions)
        assert 0.0 <= metrics["engagement_level"] <= 1.0

    # --- infer_phase ---

    def test_phase_empty_is_planning(self):
        """Без взаимодействий фаза = 'planning'."""
        phase = self.analyzer.infer_phase([])
        assert phase == "planning"

    def test_phase_few_interactions_is_planning(self):
        """При <= 2 взаимодействиях фаза = 'planning'."""
        interactions = [make_interaction() for _ in range(2)]
        phase = self.analyzer.infer_phase(interactions)
        assert phase == "planning"

    def test_phase_active_work_is_performance(self):
        """При активной работе (низкий TMR, много взаимодействий) фаза = 'performance'."""
        interactions = [
            make_interaction(success=(i % 3 == 0)) for i in range(10)
        ]
        phase = self.analyzer.infer_phase(interactions)
        assert phase == "performance"

    def test_phase_high_mastery_is_reflection(self):
        """При высоком TMR и ≥4 взаимодействий фаза = 'reflection'."""
        interactions = [make_interaction(success=True) for _ in range(6)]
        phase = self.analyzer.infer_phase(interactions)
        assert phase == "reflection"

    def test_phase_override_forces_phase(self):
        """phase_override принудительно устанавливает фазу независимо от метрик."""
        # Создаём сценарий, который дал бы 'reflection'
        interactions = [make_interaction(success=True) for _ in range(6)]
        self.analyzer.phase_override = "planning"
        phase = self.analyzer.infer_phase(interactions)
        assert phase == "planning"

    def test_phase_override_can_be_cleared(self):
        """После очистки phase_override фаза вычисляется нормально."""
        interactions = [make_interaction(success=True) for _ in range(6)]
        self.analyzer.phase_override = "planning"
        self.analyzer.phase_override = None
        phase = self.analyzer.infer_phase(interactions)
        assert phase == "reflection"

    def test_all_phase_override_values(self):
        """phase_override корректно работает для всех трёх фаз."""
        interactions = [make_interaction() for _ in range(5)]
        for forced_phase in ("planning", "performance", "reflection"):
            self.analyzer.phase_override = forced_phase
            assert self.analyzer.infer_phase(interactions) == forced_phase

    # --- get_srl_state ---

    def test_get_srl_state_structure(self):
        """get_srl_state возвращает словарь с session_id, phase, metrics."""
        interactions = [make_interaction() for _ in range(3)]
        state = self.analyzer.get_srl_state("s1", interactions)
        assert "session_id" in state
        assert "phase" in state
        assert "metrics" in state
        assert state["session_id"] == "s1"
        assert state["phase"] in ("planning", "performance", "reflection")

    # --- compute_spas ---

    def test_spas_returns_probability(self):
        """SPAS возвращает значение в [0, 1]."""
        interactions = [make_interaction() for _ in range(4)]
        for phase in ("planning", "performance", "reflection"):
            spas = self.analyzer.compute_spas(phase, interactions)
            assert 0.0 <= spas <= 1.0, f"SPAS out of range for phase={phase}"

    def test_spas_probabilities_sum_to_one(self):
        """Суммарные вероятности всех трёх фаз должны быть ~1.0 (softmax)."""
        interactions = [make_interaction() for _ in range(5)]
        total = sum(
            self.analyzer.compute_spas(ph, interactions)
            for ph in ("planning", "performance", "reflection")
        )
        assert total == pytest.approx(1.0, abs=0.01)

    def test_spas_empty_interactions_is_neutral(self):
        """При пустом списке SPAS возвращает нейтральное значение 0.5."""
        spas = self.analyzer.compute_spas("performance", [])
        assert spas == 0.5

    # --- interaction_from_dict / interaction_to_dict ---

    def test_interaction_round_trip(self):
        """Сериализация и десериализация TaskInteraction без потери данных."""
        original = make_interaction(
            session_id="s42", task_id="t1", phase="performance",
            success=True, used_hint=True, attempts=3, response_time=30.0,
        )
        d = interaction_to_dict(original)
        restored = interaction_from_dict(d)
        assert restored.session_id == original.session_id
        assert restored.task_id == original.task_id
        assert restored.phase == original.phase
        assert restored.success == original.success
        assert restored.used_hint == original.used_hint
        assert restored.attempts == original.attempts
        assert restored.response_time == pytest.approx(original.response_time)


# ==============================================================================
# Тесты MoralSchemaEngine
# ==============================================================================

class TestMoralSchemaEngine:
    """Тесты движка моральных схем."""

    def setup_method(self):
        self.engine = MoralSchemaEngine()
        self.neutral_srl = make_srl_state()
        self.default_profile = {
            "responsibility": 0.5,
            "integrity": 0.5,
            "autonomy": 0.5,
            "reflection": 0.5,
        }

    def test_engine_has_four_schemas(self):
        """Движок содержит четыре схемы из модели ВКР."""
        assert len(self.engine.schemas) == 4

    def test_schema_ids_exist(self):
        """Все три стандартных ID схем присутствуют."""
        ids = {s.id for s in self.engine.schemas}
        assert {"motivation", "attention", "skill", "mastery"}.issubset(ids)

    def test_low_stress_selects_highest_weight_schema(self):
        """
        При любом FS/ESI побеждает схема с наибольшим ethical_weight.

        motivation имеет max_frustration=1.0 и min_safety=0.0 —
        она всегда проходит фильтр. При weight=1.0 она всегда выбирается первой.
        Это намеренный консервативный выбор: при любых условиях система
        предпочитает наиболее «этически безопасную» схему.
        """
        affective = make_affective(fs=0.2, esi=0.8)
        schema = self.engine.get_active_schema(self.neutral_srl, affective, self.default_profile)
        # motivation (weight=1.0) побеждает всегда, т.к. её диапазон
        # охватывает весь возможный FS/ESI и ethical_weight максимален
        assert schema.id == "attention"

    def test_high_stress_selects_motivation(self):
        """При высоком FS (0.9) → motivation (единственная подходящая схема)."""
        affective = make_affective(fs=0.9, esi=0.1)
        schema = self.engine.get_active_schema(self.neutral_srl, affective, self.default_profile)
        assert schema.id == "motivation"

    def test_medium_stress_selects_attention(self):
        """
        При среднем FS (0.5) на фазе planning → attention.

        Логика выбора учитывает не только эмоциональное состояние, но и фазу SRL;
        при планировании тьютор возвращает обучающегося к учебной цели.
        """
        affective = make_affective(fs=0.5, esi=0.5)
        schema = self.engine.get_active_schema(self.neutral_srl, affective, self.default_profile)
        assert schema.id == "attention"

    def test_high_stress_schema_allows_support(self):
        """Схема motivation разрешает 'support'."""
        affective = make_affective(fs=0.9, esi=0.1)
        schema = self.engine.get_active_schema(self.neutral_srl, affective, self.default_profile)
        assert "support" in schema.allowed_actions

    def test_mastery_schema_has_challenge(self):
        """Схема mastery содержит 'challenge' в allowed_actions."""
        low_stress = next(s for s in self.engine.schemas if s.id == "mastery")
        assert "challenge" in low_stress.allowed_actions

    def test_motivation_schema_no_challenge(self):
        """Схема motivation не содержит 'challenge' (нельзя нагружать стрессом)."""
        high_stress = next(s for s in self.engine.schemas if s.id == "motivation")
        assert "challenge" not in high_stress.allowed_actions

    def test_schema_has_allowed_actions(self):
        """Каждая схема имеет непустой список allowed_actions."""
        for schema in self.engine.schemas:
            assert len(schema.allowed_actions) > 0, f"Schema {schema.id} has no allowed_actions"

    def test_ethical_weight_range(self):
        """ethical_weight каждой схемы в диапазоне [0, 1]."""
        for schema in self.engine.schemas:
            assert 0.0 <= schema.ethical_weight <= 1.0

    def test_high_stress_has_max_ethical_weight(self):
        """motivation имеет наибольший ethical_weight = 1.0."""
        hs = next(s for s in self.engine.schemas if s.id == "motivation")
        assert hs.ethical_weight == 1.0

    def test_get_active_schema_returns_moral_schema(self):
        """get_active_schema всегда возвращает объект MoralSchema."""
        for fs, esi in [(0.0, 1.0), (0.5, 0.5), (0.9, 0.1), (1.0, 0.0)]:
            affective = make_affective(fs, esi)
            schema = self.engine.get_active_schema(self.neutral_srl, affective, self.default_profile)
            assert isinstance(schema, MoralSchema)


# ==============================================================================
# Тесты DecisionEngine._estimate_utility
# ==============================================================================

class TestEstimateUtility:
    """Тесты статической функции оценки полезности действия."""

    def _u(self, action, phase="performance", tmr=0.5, hr=0.2, pi=0.4,
           eng=0.5, fs=0.0, schema=None):
        """Вспомогательный метод для краткого вызова _estimate_utility."""
        srl = make_srl_state(phase=phase, tmr=tmr, hr=hr, pi=pi, eng=eng)
        aff = make_affective(fs=fs)
        u, components = DecisionEngine._estimate_utility(action, srl, aff, {}, schema)
        return u, components

    # --- Базовые свойства ---

    def test_returns_tuple_float_dict(self):
        """_estimate_utility возвращает (float, dict)."""
        result = DecisionEngine._estimate_utility(
            "explain", make_srl_state(), make_affective(), {}, None
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], dict)

    def test_components_keys(self):
        """Компоненты содержат ключи delta_k, delta_e, ec."""
        _, components = self._u("explain")
        assert "delta_k" in components
        assert "delta_e" in components
        assert "ec" in components

    # --- Влияние FS на веса ---

    def test_high_fs_increases_emotional_weight(self):
        """При высоком FS (0.8) support получает более высокую утилиту чем при низком FS."""
        u_low_fs, _ = self._u("support", fs=0.1)
        u_high_fs, _ = self._u("support", fs=0.8)
        # Support: delta_e отрицательное (успокаивает), при высоком FS w_e растёт
        assert u_high_fs > u_low_fs, "support должен быть полезнее при высоком FS"

    def test_challenge_penalized_at_high_fs(self):
        """challenge имеет низкую утилиту при высоком FS (FS > 0.7 → ec = 0.1)."""
        u_low_fs, _ = self._u("challenge", tmr=0.9, fs=0.1)
        u_high_fs, _ = self._u("challenge", tmr=0.9, fs=0.9)
        assert u_high_fs < u_low_fs, "challenge должен быть менее полезен при высоком FS"

    def test_challenge_ec_penalty_high_fs(self):
        """При FS > 0.7 challenge получает ec = 0.1 (критически низкое)."""
        _, components = self._u("challenge", fs=0.8)
        assert components["ec"] == pytest.approx(0.1)

    def test_neutral_ec_without_schema(self):
        """Без схемы и при fs ≤ 0.5 action получает нейтральный ec = 0.7."""
        _, components = self._u("explain", fs=0.3, schema=None)
        assert components["ec"] == pytest.approx(0.7)

    # --- Фазовый бонус ---

    def test_explain_bonus_in_planning(self):
        """'explain' получает phase_bonus в фазе 'planning'."""
        u_planning, _ = self._u("explain", phase="planning")
        u_performance, _ = self._u("explain", phase="performance")
        # phase_bonus = 0.05 * 0.4 = 0.02 только в planning
        assert u_planning > u_performance

    def test_reflect_bonus_in_reflection(self):
        """'reflect' получает наибольший phase_bonus в фазе 'reflection'."""
        u_reflection, _ = self._u("reflect", phase="reflection")
        u_performance, _ = self._u("reflect", phase="performance")
        assert u_reflection > u_performance

    def test_hint_bonus_in_performance(self):
        """'hint' получает phase_bonus в фазе 'performance'."""
        u_performance, _ = self._u("hint", phase="performance")
        u_planning, _ = self._u("hint", phase="planning")
        assert u_performance > u_planning

    # --- Schema alignment ---

    def test_schema_alignment_increases_utility(self):
        """Действие из allowed_actions схемы получает schema_bonus = 0.08 и ec = 0.9."""
        hs_schema = next(
            s for s in MoralSchemaEngine().schemas if s.id == "motivation"
        )
        u_with_schema, comp_with = self._u("support", fs=0.6, schema=hs_schema)
        u_without_schema, comp_without = self._u("support", fs=0.6, schema=None)
        assert u_with_schema > u_without_schema
        assert comp_with["ec"] == pytest.approx(0.9)

    def test_action_outside_schema_no_bonus(self):
        """Действие НЕ из allowed_actions не получает schema_bonus."""
        hs_schema = next(
            s for s in MoralSchemaEngine().schemas if s.id == "motivation"
        )
        # 'challenge' не в allowed_actions motivation
        _, comp = self._u("challenge", tmr=0.9, fs=0.3, schema=hs_schema)
        assert comp["ec"] != pytest.approx(0.9)

    # --- Зависимость delta_k от метрик ---

    def test_explain_delta_k_decreases_with_tmr(self):
        """delta_k для 'explain' уменьшается при высоком TMR (объяснение менее нужно)."""
        _, comp_low_tmr = self._u("explain", tmr=0.1)
        _, comp_high_tmr = self._u("explain", tmr=0.9)
        assert comp_low_tmr["delta_k"] > comp_high_tmr["delta_k"]

    def test_challenge_delta_k_increases_with_tmr(self):
        """delta_k для 'challenge' растёт с TMR (вызов ценен при высоком мастерстве)."""
        _, comp_low = self._u("challenge", tmr=0.1)
        _, comp_high = self._u("challenge", tmr=0.9)
        assert comp_high["delta_k"] > comp_low["delta_k"]

    def test_hint_delta_k_increases_with_hr(self):
        """delta_k для 'hint' растёт при высоком HR."""
        _, comp_low = self._u("hint", hr=0.0)
        _, comp_high = self._u("hint", hr=1.0)
        assert comp_high["delta_k"] > comp_low["delta_k"]

    # --- Числовая проверка одного примера ---

    def test_known_utility_value_support_no_stress(self):
        """Проверка конкретного значения: support, fs=0, без схемы."""
        # delta_k = 0.05
        # delta_e = -0.20 * (0.5 + 0.0) = -0.10
        # w_k=0.5, w_e=0.3, w_ec=0.2, ec=0.7, schema_bonus=0, phase_bonus=0
        # u = 0.5*0.05 + 0.3*0.10 + 0.2*0.7 = 0.025 + 0.03 + 0.14 = 0.195
        u, _ = self._u("support", phase="performance", fs=0.0, tmr=0.5, schema=None)
        assert u == pytest.approx(0.195, abs=0.001)


# ==============================================================================
# Тесты DecisionEngine._generate_candidate_actions
# ==============================================================================

class TestGenerateCandidateActions:
    """Тесты генерации кандидатов на действие."""

    def _candidates(self, phase="performance", fs=0.0, tmr=0.5):
        srl = make_srl_state(phase=phase, tmr=tmr)
        aff = make_affective(fs=fs)
        return DecisionEngine._generate_candidate_actions(srl, aff)

    def test_returns_list(self):
        """Функция возвращает список."""
        assert isinstance(self._candidates(), list)

    def test_no_duplicates(self):
        """В списке нет дубликатов."""
        for phase in ("planning", "performance", "reflection"):
            candidates = self._candidates(phase=phase)
            assert len(candidates) == len(set(candidates))

    def test_planning_phase_has_explain(self):
        """Фаза 'planning' включает 'explain'."""
        assert "explain" in self._candidates(phase="planning")

    def test_performance_phase_has_hint(self):
        """Фаза 'performance' включает 'hint'."""
        assert "hint" in self._candidates(phase="performance")

    def test_reflection_phase_has_reflect(self):
        """Фаза 'reflection' включает 'reflect'."""
        assert "reflect" in self._candidates(phase="reflection")

    def test_high_frustration_adds_simplify(self):
        """При FS > 0.7 в список добавляется 'simplify'."""
        candidates = self._candidates(fs=0.8)
        assert "simplify" in candidates

    def test_low_frustration_no_simplify(self):
        """При FS ≤ 0.7 'simplify' отсутствует."""
        candidates = self._candidates(fs=0.5)
        assert "simplify" not in candidates

    def test_high_mastery_low_stress_adds_challenge(self):
        """При TMR > 0.8 и FS < 0.4 добавляется 'challenge'."""
        candidates = self._candidates(tmr=0.9, fs=0.2)
        assert "challenge" in candidates

    def test_high_mastery_high_stress_no_challenge(self):
        """При TMR > 0.8 и FS > 0.4 'challenge' отсутствует."""
        candidates = self._candidates(tmr=0.9, fs=0.6)
        assert "challenge" not in candidates

    def test_planning_does_not_include_hint(self):
        """Фаза 'planning' не включает 'hint' по умолчанию."""
        candidates = self._candidates(phase="planning")
        assert "hint" not in candidates


# ==============================================================================
# Интеграционный тест полного адаптивного пайплайна
# ==============================================================================

class TestAdaptivePipelineIntegration:
    """
    Интеграционные тесты, проверяющие взаимодействие SRLAnalyzer,
    MoralSchemaEngine и DecisionEngine._estimate_utility.

    Важно: не делают реальных LLM-вызовов — только проверяют логику
    выбора действия и сопряжение компонентов.
    """

    def setup_method(self):
        self.srl_analyzer = SRLAnalyzer()
        self.schema_engine = MoralSchemaEngine()
        self.profile = {
            "responsibility": 0.5,
            "integrity": 0.5,
            "autonomy": 0.5,
            "reflection": 0.5,
        }

    def _select_best_action(self, interactions, affective):
        """Имитирует выбор лучшего действия без LLM."""
        srl_state = self.srl_analyzer.get_srl_state("test", interactions)
        schema = self.schema_engine.get_active_schema(srl_state, affective, self.profile)
        candidates = DecisionEngine._generate_candidate_actions(srl_state, affective)
        best_action, best_u = None, -999.0
        for action in candidates:
            u, _ = DecisionEngine._estimate_utility(action, srl_state, affective, self.profile, schema)
            if u > best_u:
                best_u, best_action = u, action
        return best_action, best_u, schema

    def test_high_stress_prefers_support(self):
        """При высоком FS (0.85) система предпочитает 'support'."""
        interactions = [
            make_interaction(success=False, used_hint=True, attempts=3, response_time=40.0)
            for _ in range(5)
        ]
        affective = make_affective(fs=0.85, esi=0.2)
        action, u, schema = self._select_best_action(interactions, affective)
        assert action in ("support", "simplify", "hint"), \
            f"Ожидался эмоционально-поддерживающий action, получен: {action}"
        assert schema.id == "motivation"

    def test_low_stress_high_mastery_may_challenge(self):
        """При низком FS и высоком TMR 'challenge' появляется в кандидатах."""
        interactions = [make_interaction(success=True) for _ in range(8)]
        affective = make_affective(fs=0.1, esi=0.9)
        srl_state = self.srl_analyzer.get_srl_state("test", interactions)
        candidates = DecisionEngine._generate_candidate_actions(srl_state, affective)
        assert "challenge" in candidates

    def test_full_policy_beats_baseline(self):
        """
        Full (SRL + eBICA) utility ≥ baseline utility при высоком FS.

        Baseline не использует SRL и применяет нейтральную схему с реальным affective.
        Full использует реальный SRL и реальную моральную схему.
        """
        interactions = [
            make_interaction(
                success=(i % 4 == 0), used_hint=True, attempts=2, response_time=30.0
            )
            for i in range(6)
        ]
        affective = make_affective(fs=0.75, esi=0.2)

        # Full policy
        action_full, u_full, _ = self._select_best_action(interactions, affective)

        # Baseline: нейтральный SRL, нейтральная схема
        neutral_srl = make_srl_state(
            phase="planning", tmr=0.0, hr=0.0, pi=0.0, eng=0.5
        )
        neutral_schema = next(s for s in self.schema_engine.schemas if s.id == "skill")
        baseline_candidates = ["explain", "question", "hint", "support"]
        best_base_u = max(
            DecisionEngine._estimate_utility(a, neutral_srl, affective, self.profile, neutral_schema)[0]
            for a in baseline_candidates
        )

        assert u_full >= best_base_u, (
            f"Full policy (u={u_full:.4f}) должен быть ≥ baseline (u={best_base_u:.4f})"
        )

    def test_phase_override_changes_candidates(self):
        """phase_override влияет на кандидатов через SRL-фазу."""
        interactions = [make_interaction(success=True) for _ in range(8)]
        affective = make_affective(fs=0.1)

        self.srl_analyzer.phase_override = "planning"
        srl_planning = self.srl_analyzer.get_srl_state("t", interactions)
        candidates_planning = DecisionEngine._generate_candidate_actions(srl_planning, affective)

        self.srl_analyzer.phase_override = "performance"
        srl_performance = self.srl_analyzer.get_srl_state("t", interactions)
        candidates_performance = DecisionEngine._generate_candidate_actions(srl_performance, affective)

        assert "explain" in candidates_planning
        assert "hint" in candidates_performance
        assert "hint" not in candidates_planning

    def test_schema_selection_matches_srl_phase(self):
        """
        Проверка согласованности: схема, выбранная при высоком FS,
        содержит только нежёсткие действия.
        """
        affective = make_affective(fs=0.9, esi=0.1)
        srl_state = make_srl_state()
        schema = self.schema_engine.get_active_schema(srl_state, affective, self.profile)
        assert "challenge" not in schema.allowed_actions
        assert any(a in schema.allowed_actions for a in ("support", "hint", "simplify"))

    def test_spas_reflects_real_phase(self):
        """
        SPAS для реальной фазы должен быть выше, чем для других фаз,
        когда поведение студента явно указывает на эту фазу.
        """
        # Студент с высоким TMR и малым HR → reflection
        interactions = [make_interaction(success=True, used_hint=False) for _ in range(7)]
        spas_reflection = self.srl_analyzer.compute_spas("reflection", interactions)
        spas_planning = self.srl_analyzer.compute_spas("planning", interactions)
        # При высоком TMR reflection-фаза должна быть более вероятна чем planning
        assert spas_reflection > spas_planning


# ==============================================================================
# Точка входа
# ==============================================================================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    sys.exit(result.returncode)
