import numpy as np
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
from app.helper import load_config
from typing import Generator
from functools import lru_cache

class Interface:
    """Класс для взаимодействия с DeepSeek API"""
    
    def __init__(self, token: str = None, base_url: str = None, model_name: str = None):
        if token:
            self.token = token
            self.base_url = (base_url or "https://api.deepseek.com").rstrip('/')
        else:
            config = load_config()
            self.token = config["auth"]["token"]
            self.base_url = config["auth"]["url"].rstrip('/')
        self.model_name = model_name or ("gpt-4o-mini" if "openai.com" in self.base_url else "deepseek-chat")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        # Параметры сети
        self.timeout = (10, 60)  # (connect timeout, read timeout)
        self.max_retries = 3
        self.backoff_delay = 0.5

        # Постоянная сессия с пулом соединений и ретраями для сокращения задержек на установку TCP/TLS
        self.session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            backoff_factor=self.backoff_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=50, max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # Кэш для быстрых повторов AI-рекомендаций
        self._last_ai_rec_key = None
        self._last_ai_rec_value = None
        self._last_ai_rec_time = 0.0

    def _make_api_request(self, endpoint, body):
        url = f"{self.base_url}{endpoint}"
        body = dict(body or {})
        # Все внутренние модули используют модель активного провайдера сессии.
        body["model"] = self.model_name
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    url=url,
                    headers=self.headers,
                    json=body,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                print(f"⚠️ API timeout (attempt {attempt+1}/{self.max_retries}): {url}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.backoff_delay)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                body_text = (e.response.text[:300] if e.response is not None else "")
                print(f"⚠️ API HTTP {status} (attempt {attempt+1}/{self.max_retries}): {body_text}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.backoff_delay)
            except requests.exceptions.RequestException as e:
                print(f"⚠️ API request error ({type(e).__name__}, attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.backoff_delay)
        return None

    def _stream_api_request(self, endpoint: str, body: dict) -> Generator[str, None, None]:
        """Выполняет streaming-запрос к API и построчно возвращает дельты контента.

        Ожидается SSE-формат, совместимый с OpenAI: строки вида "data: {json}"
        с полем choices[0].delta.content. Завершается на строке 'data: [DONE]'.
        """
        url = f"{self.base_url}{endpoint}"
        # Добавляем флаг stream
        body_stream = dict(body)
        body_stream["model"] = self.model_name
        body_stream["stream"] = True

        for attempt in range(self.max_retries):
            try:
                with self.session.post(
                    url=url,
                    headers=self.headers,
                    json=body_stream,
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data:"):
                            payload = line[len("data:"):].strip()
                            if payload == "[DONE]":
                                return
                            try:
                                obj = json.loads(payload)
                                # Пытаемся извлечь дельту в формате OpenAI
                                choices = obj.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    content_piece = delta.get("content")
                                    if content_piece:
                                        yield content_piece
                                    # DeepSeek может отдавать и полные сообщения
                                    msg = choices[0].get("message")
                                    if msg and msg.get("content"):
                                        yield msg.get("content")
                            except Exception:
                                # В случае нестандартной строки — пропускаем
                                continue
                    return
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    return
                time.sleep(self.backoff_delay)
            except requests.exceptions.RequestException:
                if attempt == self.max_retries - 1:
                    return
                time.sleep(self.backoff_delay)
        return

    @staticmethod
    def _finalize_response_text(text: str) -> str:
        """Минимальная пост-обработка для аккуратного вывода.
        - Закрывает незавершённые блоки кода ```
        - Гарантирует завершающую пунктуацию
        """
        try:
            if text is None:
                return text
            # Закрываем незакрытый код-блок
            fence_count = text.count('```')
            if fence_count % 2 == 1:
                text += "\n```"
            # Добавляем точку, если текст обрывается без знака
            stripped = text.rstrip()
            if stripped and stripped[-1] not in '.!?…' and not stripped.endswith('```'):
                text = stripped + '.'
            return text
        except Exception:
            return text

    def clear_intentions(self, reply):
        """Извлечение числовых значений из ответа модели"""
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", reply)
        return [float(num) if '.' in num else int(num) for num in numbers]

    @lru_cache(maxsize=256)
    def _cached_composition(self, cat_str: str, phrase: str):
        prompt = f"""
        Ты механизм по определению интенций в речи человека.
        Доступные категории: {cat_str}
        
        Фраза пользователя: "{phrase}"
        
        Проанализируй фразу и верни вектор числовых значений, отражающих степень проявления каждой интенции.
        Используй значения от -1 до 1, где:
        -1: сильное отрицательное проявление
        0: нейтральное
        1: сильное положительное проявление
        
        Верни только числа, разделенные запятыми.
        """
        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 100
        }
        result = self._make_api_request("/chat/completions", body)
        if result:
            return self.clear_intentions(result["choices"][0]["message"]["content"]) or []
        return []

    def get_composition(self, intents_dict, phrase):
        """Получение вектора действий на основе интенций"""
        cat_str = ', '.join(intents_dict.values())
        try:
            vals = self._cached_composition(cat_str, phrase)
            return vals if vals else None
        except Exception:
            pass
        # Fallback без кэша
        prompt = f"""
        Ты механизм по определению интенций в речи человека.
        Доступные категории: {cat_str}
        
        Фраза пользователя: "{phrase}"
        
        Проанализируй фразу и верни вектор числовых значений, отражающих степень проявления каждой интенции.
        Используй значения от -1 до 1, где:
        -1: сильное отрицательное проявление
        0: нейтральное
        1: сильное положительное проявление
        
        Верни только числа, разделенные запятыми.
        """
        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.1,
            'max_tokens': 100
        }
        result = self._make_api_request("/chat/completions", body)
        if result:
            return self.clear_intentions(result["choices"][0]["message"]["content"])
        return None

    def get_replic(self, last_message, messages, intens_dict, feelings, prev_scheme, current_scheme, adaptive_context=""):
        """Генерация адаптивного ответа тьютора"""
        rlt = [(intens_dict[i if val > -0.05 else -i], val) 
              for i, val in enumerate(feelings, start=1)]
        
        # Анализ профиля студента
        student_profile = "Характеристика студента:\n"
        for idx, (trait, value) in enumerate(rlt, start=1):
            student_profile += f"{idx}. {trait}: {value:.2f}\n"
            if value < -0.05:
                student_profile += f"   ⚠️ Требуется коррекция: {trait}\n"

        # Определение переходного сообщения
        transition_msg = ""
        if current_scheme - prev_scheme == 1:
            if current_scheme == 2:
                transition_msg = "🎉 Согласие начать урок получено. Переходим к изучению рекурсии!"
            elif current_scheme == 3:
                transition_msg = "🎯 Ученик успешно освоил рекурсию. Переходим к изучению инкапсуляции!"
            elif current_scheme == 4:
                transition_msg = "🏆 Задания выполнены! Переходим к финальной оценке знаний."
        else:
            stage_names = ["Мотивация", "Внимание", "Навык", "Мастерство"]
            transition_msg = f"📍 Вы находитесь на этапе: {stage_names[current_scheme]}"

        # Формирование адаптивного промпта
        adaptive_instructions = ""
        if adaptive_context:
            adaptive_instructions = f"""
            {adaptive_context}
            
            Инструкции по адаптации:
            - Если уровень сложности высокий (4-5), давай более сложные задачи
            - Если детализация объяснений высокая (3), давай подробные пояснения
            - Если соотношение практики > 0.8, фокусируйся на практических заданиях
            - Если точность студента < 0.6, давай больше подсказок и разъяснений
            """

        prompt = f"""
        {student_profile}
        
        {transition_msg}
        
        {adaptive_instructions}
        
        Последняя реплика студента: "{last_message}"
        
        Сгенерируй педагогически выверенный ответ (до 150 слов), учитывая:
        1. Профиль студента и его текущие потребности
        2. Этап обучения и переходные моменты
        3. Адаптивные настройки для персонализации
        4. Метод Сократа (наводящие вопросы)
        5. Поддержку мотивации и позитивного настроя
        
        Используй Markdown для форматирования кода и важных моментов.
        """

        # Ограничиваем историю для скорости (последние 6 сообщений + системный)
        base = messages[:1] if messages and messages[0].get('role') in ('system', 'assistant') else []
        tail = messages[-6:] if len(messages) > 6 else messages
        messages_opt = base + tail + [{"role": "user", "content": prompt}]

        # Адаптивные параметры генерации
        temperature = 0.5 if "обсуждение" in last_message.lower() else 0.45
        # Чуть увеличиваем лимит, чтобы не обрывались окончания
        max_tokens = 260 if adaptive_context else 220

        body = {
            'model': self.model_name,
            'messages': messages_opt,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        result = self._make_api_request("/chat/completions", body)
        if result:
            content = result["choices"][0]["message"]["content"]
            return self._finalize_response_text(content)
        return "Не удалось получить ответ от API"

    def stream_replic(self, last_message, messages, intens_dict, feelings, prev_scheme, current_scheme, adaptive_context="") -> Generator[str, None, None]:
        """Потоковая генерация ответа тьютора. Возвращает генератор дельт текста."""
        rlt = [(intens_dict[i if val > -0.05 else -i], val) 
              for i, val in enumerate(feelings, start=1)]

        student_profile = "Характеристика студента:\n"
        for idx, (trait, value) in enumerate(rlt, start=1):
            student_profile += f"{idx}. {trait}: {value:.2f}\n"
            if value < -0.05:
                student_profile += f"   ⚠️ Требуется коррекция: {trait}\n"

        transition_msg = ""
        if current_scheme - prev_scheme == 1:
            if current_scheme == 2:
                transition_msg = "🎉 Согласие начать урок получено. Переходим к изучению рекурсии!"
            elif current_scheme == 3:
                transition_msg = "🎯 Ученик успешно освоил рекурсию. Переходим к изучению инкапсуляции!"
            elif current_scheme == 4:
                transition_msg = "🏆 Задания выполнены! Переходим к финальной оценке знаний."
        else:
            stage_names = ["Мотивация", "Внимание", "Навык", "Мастерство"]
            transition_msg = f"📍 Вы находитесь на этапе: {stage_names[current_scheme]}"

        adaptive_instructions = ""
        if adaptive_context:
            adaptive_instructions = f"""
            {adaptive_context}
            
            Инструкции по адаптации:
            - Если уровень сложности высокий (4-5), давай более сложные задачи
            - Если детализация объяснений высокая (3), давай подробные пояснения
            - Если соотношение практики > 0.8, фокусируйся на практических заданиях
            - Если точность студента < 0.6, давай больше подсказок и разъяснений
            """

        prompt = f"""
        {student_profile}
        
        {transition_msg}
        
        {adaptive_instructions}
        
        Последняя реплика студента: "{last_message}"
        
        Сгенерируй педагогически выверенный ответ (до 150 слов), учитывая:
        1. Профиль студента и его текущие потребности
        2. Этап обучения и переходные моменты
        3. Адаптивные настройки для персонализации
        4. Метод Сократа (наводящие вопросы)
        5. Поддержку мотивации и позитивного настроя
        
        Используй Markdown для форматирования кода и важных моментов.
        """

        base = messages[:1] if messages and messages[0].get('role') in ('system', 'assistant') else []
        tail = messages[-6:] if len(messages) > 6 else messages
        messages_opt = base + tail + [{"role": "user", "content": prompt}]

        temperature = 0.5 if "обсуждение" in last_message.lower() else 0.45
        max_tokens = 260 if adaptive_context else 220

        body = {
            'model': self.model_name,
            'messages': messages_opt,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        for delta in self._stream_api_request("/chat/completions", body):
            if delta:
                yield delta

    def generate_tail_completion(self, partial_text: str, max_tokens: int = 60) -> str:
        """Быстро завершает незаконченную фразу без повторов исходного текста.

        Возвращает только продолжение. Старается закончить на ближайшей точке/восклицании/вопросе.
        """
        try:
            prompt = (
                "Продолжи следующий текст естественно на русском, закончив последнюю фразу.\n"
                "Верни ТОЛЬКО продолжение без повторения исходного текста, 1-2 предложения максимум.\n\n"
                f"Текст:\n{partial_text}\n\nПродолжение:"
            )
            body = {
                'model': self.model_name,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.2,
                'max_tokens': max_tokens
            }
            result = self._make_api_request("/chat/completions", body)
            if not result:
                return ""
            cont = result["choices"][0]["message"]["content"] or ""
            # Усечем по первому завершению предложения
            for end in ['. ', '! ', '? ', '.\n', '!\n', '?\n', '.', '!', '?']:
                idx = cont.find(end)
                if idx != -1:
                    cont = cont[: idx + len(end.strip())]
                    break
            return cont.strip()
        except Exception:
            return ""

    @lru_cache(maxsize=64)
    def _cached_ai_recommendations(self, cache_key: str) -> dict:
        """Кэшируемая генерация AI-рекомендаций по ключу."""
        try:
            data = json.loads(cache_key)
        except Exception:
            data = {}
        prompt = f"""
Ты — педагогический аналитик. На основе метрик обучения, текущего этапа и краткой истории
сформируй рекомендации для улучшения процесса. Ответ строго в JSON со следующими полями:
{{
  "analysis": "1 короткое предложение анализа",
  "priority_recommendations": ["до 2 пунктов"],
  "learning_recommendations": ["до 4 пунктов"],
  "motivational_advice": "1 короткая фраза поддержки",
  "next_steps": ["до 3 шагов"],
  "estimated_improvement_time": "примерный срок в часах/днях",
  "confidence_level": 0.0-1.0
}}

Контекст (метрики, округлённые):
{data.get('metrics', {})}

Адаптивные настройки: {data.get('adaptive_settings', {})}
Текущий этап (0..3): {data.get('current_stage', 0)}
Краткая история (до 3 записей): {data.get('history', [])}
"""
        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens': 220
        }
        result = self._make_api_request("/chat/completions", body)
        if not result:
            return {}
        try:
            content = result["choices"][0]["message"]["content"]
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return {}

    def generate_ai_recommendations(self, metrics: dict, adaptive_settings: dict, current_stage: int, interaction_history: list) -> dict:
        """AI-рекомендации для прогресса. Быстро, с кэшем; не влияет на скорость стрима.

        - Округляет метрики, сокращает историю
        - Кэширует по ключу, чтобы повторные запросы не вызывали модель
        """
        try:
            # Подготавливаем компактные данные для стабильного ключа
            m = metrics or {}
            compact_metrics = {
                'accuracy': round(float(m.get('accuracy', 0.0)), 3),
                'hint_rate': round(float(m.get('hint_rate', 0.0)), 3),
                'response_time': round(float(m.get('response_time', 0.0)), 2),
                'emotional_distance': round(float(m.get('emotional_distance', 0.0)), 3),
                'tests_passed': int(m.get('tests_passed', 0)),
                'total_tests': int(m.get('total_tests', 0)),
                'hints_used': int(m.get('hints_used', 0)),
            }
            compact_settings = {
                'difficulty_level': int(adaptive_settings.get('difficulty_level', 1)),
                'explanation_detail': int(adaptive_settings.get('explanation_detail', 2)),
                'hint_frequency': int(adaptive_settings.get('hint_frequency', 1)),
                'practice_ratio': round(float(adaptive_settings.get('practice_ratio', 0.8)), 2),
            }
            short_history = (interaction_history or [])[-3:]
            key_obj = {
                'metrics': compact_metrics,
                'adaptive_settings': compact_settings,
                'current_stage': int(current_stage or 0),
                'history': short_history,
            }
            cache_key = json.dumps(key_obj, ensure_ascii=False, sort_keys=True)
            return self._cached_ai_recommendations(cache_key) or {}
        except Exception:
            return {}

    

    def get_dummy_replic(self, messages):
        """Генерация ответа для тестового режима"""
        body = {
            'model': self.model_name,
            'messages': messages,
            'temperature': 0.7,
            'max_tokens': 300
        }
        
        result = self._make_api_request("/chat/completions", body)
        if result:
            return result["choices"][0]["message"]["content"]
        return "Ошибка соединения с API"

    def get_brain_status(self, messages, last_message, current_scheme):
        """Определение готовности к переходу на следующий этап"""
        stage_prompts = {
            0: f"Студент сказал: '{last_message}'. Это явное согласие начать занятие? Ответь только 'да' или 'нет'.",
            1: f"Студент выполнил задание: '{last_message}'. Это корректное решение задачи по рекурсии? Ответь только 'да' или 'нет'.",
            2: f"Студент выполнил задание: '{last_message}'. Это корректное решение задачи по инкапсуляции? Ответь только 'да' или 'нет'."
        }
        
        if current_scheme not in stage_prompts:
            return None

        # Сжимаем историю для быстрых бинарных ответов
        base = messages[:1] if messages and messages[0].get('role') in ('system', 'assistant') else []
        tail = messages[-4:] if len(messages) > 4 else messages
        messages_opt = base + tail + [{"role": "user", "content": stage_prompts[current_scheme]}]

        body = {
            'model': self.model_name,
            'messages': messages_opt,
            'temperature': 0.1,
            'max_tokens': 6
        }

        result = self._make_api_request("/chat/completions", body)
        if result:
            return result["choices"][0]["message"]["content"].lower().strip()
        return None

    def evaluate_code_solution(self, code: str, task_description: str) -> dict:
        """Оценка решения кода студента"""
        prompt = f"""
        Оцени решение студента для задачи: "{task_description}"
        
        Код студента:
        ```python
        {code}
        ```
        
        Проанализируй код по следующим критериям:
        1. Корректность (правильно ли решена задача)
        2. Стиль кода (читаемость, именование переменных)
        3. Эффективность (оптимальность решения)
        4. Безопасность (отсутствие потенциальных ошибок)
        
        Верни оценку в формате JSON:
        {{
            "correctness": 0.0-1.0,
            "style": 0.0-1.0,
            "efficiency": 0.0-1.0,
            "safety": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "feedback": "конструктивная обратная связь",
            "suggestions": ["список улучшений"]
        }}
        """
        
        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
            'max_tokens': 500
        }

        result = self._make_api_request("/chat/completions", body)
        if result:
            try:
                response_text = result["choices"][0]["message"]["content"]
                # Извлекаем JSON из ответа
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                pass
        
        # Возвращаем базовую оценку при ошибке
        return {
            "correctness": 0.5,
            "style": 0.5,
            "efficiency": 0.5,
            "safety": 0.5,
            "overall_score": 0.5,
            "feedback": "Не удалось автоматически оценить код. Проверьте решение вручную.",
            "suggestions": ["Убедитесь, что код соответствует требованиям задачи"]
        }

    def generate_hint(self, task_description: str, student_attempt: str, difficulty_level: int) -> str:
        """Генерация адаптивной подсказки"""
        hint_levels = {
            1: "очень общая подсказка",
            2: "конкретная подсказка с примером",
            3: "пошаговое руководство"
        }
        
        prompt = f"""
        Задача: {task_description}
        
        Попытка студента: {student_attempt}
        
        Уровень подсказки: {hint_levels.get(difficulty_level, "общая подсказка")}
        
        Сгенерируй подсказку, которая поможет студенту найти правильное решение,
        но не даст готовый ответ. Подсказка должна быть:
        - Конструктивной и поддерживающей
        - Соответствовать уровню сложности
        - Направлять к правильному подходу
        - Не более 100 слов
        
        Начни с "💡 Подсказка:" и дай полезный совет.
        """
        
        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.6,
            'max_tokens': 200
        }

        result = self._make_api_request("/chat/completions", body)
        if result:
            return result["choices"][0]["message"]["content"]
        return "💡 Попробуйте разбить задачу на более мелкие шаги и решать их по очереди."

    def review_code_inline(self, code: str, language: str = "python") -> str:
        """Генерирует версию кода с инлайн-комментариями на русский.

        Требование пользователя: добавлять комментарии на той же строке с ошибкой/советом.
        Для Python использовать символ комментария '#'. Вернуть только код, без пояснительного текста.
        """
        comment_token = "#" if language.lower() == "python" else "//"
        prompt = f"""
Ты выступаешь в роли опытного преподавателя программирования.

Задача: пройдись по коду и добавь ИНЛАЙН-КОММЕНТАРИИ на русском языке на тех же строках, где видишь ошибки, анти-паттерны, потенциальные исключения, нарушения стиля или места для улучшения. Используй маркер комментария '{comment_token}'.

Правила:
- Возвращай ТОЛЬКО итоговый код, без пояснений до или после.
- Комментарии добавляй в КОНЦЕ соответствующих строк после пробела и маркера комментария.
- Если строка слишком длинная, допускается перенести часть выражения, но сохраняй исходную логику.
- Не переписывай решение полностью, только минимальные правки и комментарии.
- Если код корректен, добавь короткий комментарий в начале файла: '{comment_token} Код проверен: критичных ошибок не обнаружено'.

Код:
```{language}
{code}
```
Верни только итоговый код.
"""

        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens': 2000
        }

        result = self._make_api_request("/chat/completions", body)
        if not result:
            # При ошибке возвращаем исходный код без изменений
            return code

        content = result["choices"][0]["message"]["content"]
        # Пытаемся извлечь чистый код, если модель обернула в блоки
        import re as _re
        match = _re.search(r"```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```", content)
        if match:
            return match.group(1)
        return content

    def review_code_with_summary(self, code: str, language: str = "python") -> dict:
        """Возвращает и комментированный код, и краткий отчёт об изменениях (на русском).

        Формат результата:
        {
          "commented_code": "...",
          "summary": {
            "issues_found": ["..."],
            "changes_made": ["..."],
            "why": "краткое объяснение",
            "overall": "1-3 предложения об итогах ревью"
          }
        }
        """
        commented = self.review_code_inline(code, language)

        prompt = f"""
Ты провёл ревью кода и добавил инлайн-комментарии. Сформируй краткий отчёт на русском:
- Список найденных проблем (буллеты)
- Что именно изменено/добавлено (буллеты)
- Почему это было сделано (1-2 предложения)
- Общий итог ревью (1 предложение)

Верни строго JSON:
{{
  "issues_found": ["строка"],
  "changes_made": ["строка"],
  "why": "строка",
  "overall": "строка"
}}

Контекст (исходный код):
```{language}
{code}
```

Контекст (комментированный код):
```{language}
{commented}
```
"""

        body = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.2,
            'max_tokens': 600
        }

        result = self._make_api_request("/chat/completions", body)
        summary = {"issues_found": [], "changes_made": [], "why": "", "overall": ""}
        if result:
            try:
                import re as _re, json as _json
                content = result["choices"][0]["message"]["content"]
                match = _re.search(r"\{[\s\S]*\}", content)
                if match:
                    summary = _json.loads(match.group())
            except Exception:
                pass

        return {"commented_code": commented, "summary": summary}