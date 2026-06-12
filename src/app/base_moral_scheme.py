import numpy as np
from .oai_interface import Interface
from typing import Optional, Dict, List

class BaseMoralScheme:
    def __init__(
        self, 
        base_intentions: np.ndarray, 
        changed_message: Optional[str] = None, 
        appraisals: Optional[np.ndarray] = None, 
        feelings: Optional[np.ndarray] = None
    ) -> None:
        """
        Базовая моральная схема для отслеживания эмоционально-смыслового состояния студента.
        
        Args:
            base_intentions (np.ndarray): Базисные векторы семантического пространства.
            changed_message (Optional[str]): Начальный prompt для перехода к следующему этапу.
            appraisals (Optional[np.ndarray]): Вектор оценок (если не задан, заполняется нулями).
            feelings (Optional[np.ndarray]): Вектор чувств (если не задан, заполняется 0.5).
        """
        self.base_intentions = base_intentions
        self.changed_message = changed_message

        # размер семантического пространства
        self.space_size = len(base_intentions)

        # константы для формул обновления (согласно курсовой работе)
        self.p_const = 0.03  # Константа для обновления чувств
        self.r_const = 0.1   # Константа для обновления оценок

        # Интерфейс для взаимодействия с DeepSeek через прокси
        self.oai_interface = Interface()

        # Векторы состояний
        self.appraisals_state = np.zeros(self.space_size//2)
        self.feelings_state = np.zeros(self.space_size//2)

        # Инициализация векторов оценок и чувств
        if appraisals is None:
            self.appraisals = np.full(self.space_size, 0.0)
        else:
            self.appraisals = appraisals
        
        if feelings is None:
            self.feelings = np.full(self.space_size, 0.5)
        else:
            self.feelings = feelings

        # История изменений для анализа трендов
        self.appraisals_history = [self.appraisals.copy()]
        self.feelings_history = [self.feelings.copy()]
        self.max_history_size = 10

    def euc_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Вычисляет евклидово расстояние между двумя векторами.
        
        Args:
            a (np.ndarray): Первый вектор.
            b (np.ndarray): Второй вектор.
        
        Returns:
            float: Евклидово расстояние между векторами.
        
        Raises:
            ValueError: Если векторы имеют разную длину.
        """
        if a.shape != b.shape:
            raise ValueError("Векторы должны иметь одинаковую длину")
        return np.linalg.norm(a-b)
        
    def get_base_intentions(self) -> np.ndarray:
        """Возвращает базовые векторы семантического пространства."""
        return self.base_intentions 

    def update_vectors(self, action: np.ndarray):
        """
        Обновляет векторы оценок (appraisals) и чувств (feelings) 
        на основе действия (action) студента.

        Args:
            action (np.ndarray): Вектор изменений, влияющий на оценки.
        """
        # Обновление оценок (логическая реакция)
        self.appraisals = (
            (1 - self.r_const) * self.appraisals + self.r_const * action
        )
        
        # Обновление чувств (эмоциональная реакция)
        self.feelings = (
            (1 - self.p_const) * self.feelings 
            + self.p_const * (self.appraisals - self.feelings)
        )

        # Обновление состояний (разность положительных и отрицательных компонентов)
        mid = self.space_size // 2
        self.appraisals_state = self.appraisals[:mid] - self.appraisals[mid:]
        self.feelings_state = self.feelings[:mid] - self.feelings[mid:]

        # Сохранение истории
        self._update_history()

    def _update_history(self):
        """Обновление истории изменений векторов"""
        self.appraisals_history.append(self.appraisals.copy())
        self.feelings_history.append(self.feelings.copy())
        
        # Ограничение размера истории
        if len(self.appraisals_history) > self.max_history_size:
            self.appraisals_history.pop(0)
            self.feelings_history.pop(0)

    def get_appraisals(self) -> np.ndarray:
        """Возвращает вектор оценок."""
        return self.appraisals

    def get_feelings(self) -> np.ndarray:
        """Возвращает вектор чувств."""
        return self.feelings

    def get_appraisals_state(self) -> np.ndarray:
        """Возвращает текущее состояние оценок."""
        return self.appraisals_state

    def get_feelings_state(self) -> np.ndarray:
        """Возвращает текущее состояние чувств."""
        return self.feelings_state

    def get_emotional_stability(self) -> float:
        """
        Вычисляет стабильность эмоционального состояния на основе истории.
        
        Returns:
            float: Показатель стабильности (0-1, где 1 - максимальная стабильность)
        """
        if len(self.feelings_history) < 2:
            return 1.0
        
        # Вычисляем среднее изменение чувств за последние несколько шагов
        changes = []
        for i in range(1, len(self.feelings_history)):
            change = np.linalg.norm(self.feelings_history[i] - self.feelings_history[i-1])
            changes.append(change)
        
        avg_change = np.mean(changes)
        # Нормализуем к диапазону 0-1 (меньше изменений = больше стабильности)
        stability = max(0, 1 - avg_change)
        return stability

    def get_learning_momentum(self) -> float:
        """
        Вычисляет "импульс" обучения на основе тренда изменений оценок.
        
        Returns:
            float: Показатель импульса (-1 до 1, где положительные значения 
                  указывают на прогресс в обучении)
        """
        if len(self.appraisals_history) < 3:
            return 0.0
        
        # Анализируем тренд последних изменений
        recent_changes = []
        for i in range(1, len(self.appraisals_history)):
            change = np.mean(self.appraisals_history[i] - self.appraisals_history[i-1])
            recent_changes.append(change)
        
        # Вычисляем средний тренд
        momentum = np.mean(recent_changes)
        # Нормализуем к диапазону -1 до 1
        return np.clip(momentum, -1, 1)

    def get_convergence_status(self) -> Dict[str, float]:
        """
        Анализирует сходимость между оценками и чувствами.
        
        Returns:
            Dict[str, float]: Словарь с метриками сходимости
        """
        # Если ещё не было обновлений состояний, считаем, что конвергенции нет
        if len(self.appraisals_history) <= 1 or len(self.feelings_history) <= 1:
            return {
                'distance': 1.0,
                'stability': 0.0,
                'momentum': 0.0,
                'converged': False,
                'ready_for_next': False
            }

        distance = float(self.euc_dist(self.appraisals_state, self.feelings_state))
        stability = float(self.get_emotional_stability())
        momentum = float(self.get_learning_momentum())
        
        return {
            'distance': float(distance),
            'stability': float(stability),
            'momentum': float(momentum),
            'converged': bool(distance < 0.25),  # Порог сходимости из курсовой работы
            'ready_for_next': bool(distance < 0.25 and stability > 0.7 and momentum > 0.1)
        }

    def get_emotional_profile(self) -> Dict[str, float]:
        """
        Создает профиль эмоционального состояния студента.
        
        Returns:
            Dict[str, float]: Профиль с ключевыми характеристиками
        """
        # Анализируем доминирующие эмоции
        positive_emotions = float(np.sum(self.feelings[:self.space_size//2]))
        negative_emotions = float(np.sum(self.feelings[self.space_size//2:]))
        
        # Вычисляем баланс
        emotional_balance = (positive_emotions - negative_emotions) / (positive_emotions + negative_emotions + 1e-8)
        
        # Анализируем интенсивность эмоций
        emotional_intensity = float(np.std(self.feelings))
        
        return {
            'balance': float(emotional_balance),
            'intensity': float(emotional_intensity),
            'positive_ratio': float(positive_emotions / (positive_emotions + negative_emotions + 1e-8)),
            'stability': float(self.get_emotional_stability()),
            'overall_mood': 'positive' if emotional_balance > 0.1 else 'negative' if emotional_balance < -0.1 else 'neutral'
        }

    def reset_to_initial_state(self):
        """Сброс к начальному состоянию схемы"""
        self.appraisals = np.full(self.space_size, 0.0)
        self.feelings = np.full(self.space_size, 0.5)
        self.appraisals_state = np.zeros(self.space_size//2)
        self.feelings_state = np.zeros(self.space_size//2)
        self.appraisals_history = [self.appraisals.copy()]
        self.feelings_history = [self.feelings.copy()]

    def get_scheme_summary(self) -> Dict:
        """
        Получение сводной информации о состоянии моральной схемы.
        
        Returns:
            Dict: Сводка состояния схемы
        """
        convergence = self.get_convergence_status()
        emotional_profile = self.get_emotional_profile()
        
        return {
            'convergence': convergence,
            'emotional_profile': emotional_profile,
            'current_appraisals': self.appraisals_state.tolist(),
            'current_feelings': self.feelings_state.tolist(),
            'history_length': len(self.appraisals_history),
            'changed_message': self.changed_message
        }
