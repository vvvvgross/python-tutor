from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from app.data_repository import get_data_repository
from app.oai_interface import Interface
from app.helper import load_config


@dataclass
class ModelDescriptor:
    """Описатель одной LLM‑модели в оркестраторе."""

    id: str
    provider: str
    model_name: str
    # Ожидаемые характеристики (для выбора и аналитики, не жёсткие ограничения)
    expected_quality: float = 0.8
    expected_latency: float = 0.5
    expected_cost: float = 0.5


class LLMOrchestrator:
    """
    Управляемый слой поверх Interface.

    Провайдер и модель определяются текущей пользовательской сессией;
    внутренние режимы не переключают внешний API самовольно.
    """

    def __init__(self, interface: Optional[Interface] = None) -> None:
        self.interface = interface or Interface()
        self._repo = get_data_repository()
        # Используется только провайдер, выбранный пользователем для текущей сессии.
        # A/B-логика не должна незаметно переключать внешний API.
        active_provider = "openai" if "openai.com" in getattr(self.interface, "base_url", "") else "deepseek"
        active_model = getattr(self.interface, "model_name", "deepseek-chat")
        self.models: Dict[str, ModelDescriptor] = {
            "session_default": ModelDescriptor(
                id="session_default", provider=active_provider, model_name=active_model,
                expected_quality=0.9, expected_latency=0.6, expected_cost=0.5,
            ),
            "session_fast": ModelDescriptor(
                id="session_fast", provider=active_provider, model_name=active_model,
                expected_quality=0.8, expected_latency=0.4, expected_cost=0.4,
            ),
        }
        # Назначения A/B‑вариантов по session_id
        self._ab_variants: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # A/B‑тестирование
    # ------------------------------------------------------------------

    def assign_variant(self, session_id: str) -> str:
        """
        Случайно назначает вариант A или B пользователю (если ещё не назначен).
        Логирует выбор в репозиторий.
        """
        if session_id in self._ab_variants:
            return self._ab_variants[session_id]

        variant = "A" if random.random() < 0.5 else "B"
        self._ab_variants[session_id] = variant

        self._repo.log_event(
            "ab_test",
            {
                "session_id": session_id,
                "variant": variant,
                "ts": time.time(),
            },
        )
        return variant

    def _get_openai_api_key(self) -> Optional[str]:
        """
        Получает API ключ OpenAI из config.yaml или переменной окружения.
        Приоритет: config.yaml > OPENAI_API_KEY env var.
        """
        # Сначала пробуем из config.yaml
        try:
            config = load_config()
            if config and "openai" in config and "api_key" in config["openai"]:
                return config["openai"]["api_key"]
        except Exception:
            pass
        
        # Fallback на переменную окружения
        return os.environ.get("OPENAI_API_KEY")

    def select_best_model(self, context: Dict) -> ModelDescriptor:
        """
        Выбор логического режима генерации для текущего провайдера сессии.
        Провайдер не изменяется этим методом.
        """
        session_id = context.get("session_id", "unknown")
        variant = self.assign_variant(session_id)

        # Вариант влияет только на логический режим; провайдер остаётся выбранным пользователем.
        if variant == "A":
            return self.models["session_default"]
        return self.models["session_fast"]

    # ------------------------------------------------------------------
    # Генерация и оценка
    # ------------------------------------------------------------------

    def generate(self, model_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Генерация ответа через выбранную модель.

        request:
            {
              "messages": [...],
              "temperature": float,
              "max_tokens": int,
              "session_id": str,
              ...
            }
        """
        model = self.models.get(model_id, self.models["session_default"])
        messages = request.get("messages", [])
        temperature = float(request.get("temperature", 0.5))
        max_tokens = int(request.get("max_tokens", 256))

        text = ""
        result: Optional[Dict[str, Any]] = None
        latency: float = 0.0
        provider_used = model.provider
        model_used = model.id

        # Все запросы идут через Interface, созданный для текущей сессии.
        # Это предотвращает скрытую подмену DeepSeek/OpenAI и утечку ключа в другой API.
        body = {
            "model": model.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.time()
        result = self.interface._make_api_request("/chat/completions", body)  # type: ignore[attr-defined]
        latency = time.time() - started
        if result:
            try:
                text = result["choices"][0]["message"]["content"]
            except Exception:
                text = ""

        # Попытка «дотянуть» осмысленный ответ, если модель вернула пустую строку,
        # но само API доступно (result не None). Для этого используем быстрый tail‑completion
        # на последней реплике пользователя.
        if (not (text and str(text).strip())) and result:
            try:
                user_text = ""
                for m in reversed(messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        user_text = m.get("content") or ""
                        break
                if user_text:
                    tail = self.interface.generate_tail_completion(user_text)
                    if tail and tail.strip():
                        text = tail
            except Exception:
                # Если дополнительная попытка завершения не удалась, просто идём дальше
                pass

        # Глобальная страховка: если и основная генерация, и tail‑completion не дали текст,
        # считаем, что API недоступно или нестабильно, и возвращаем осмысленную заглушку.
        if not (text and str(text).strip()):
            text = (
                "Извини, сейчас сервис генерации ответов временно недоступен, "
                "и мне не удалось получить ответ от модели. "
                "Через некоторое время попробуй, пожалуйста, ещё раз."
            )

        # Формируем простую оценку ответа (Score из Q, L, C, S)
        score, score_components = self.evaluate_response(text, latency, request)

        # Логируем использование модели
        self._repo.log_event(
            "llm_usage",
            {
                "session_id": request.get("session_id"),
                "model_id": model.id,
                "provider": model.provider,
                "latency": latency,
                "score": score,
                "components": score_components,
            },
        )

        return {
            "text": text,
            "latency": latency,
            "model_id": model_used,
            "provider": provider_used,
            "raw": result,
            "score": score,
            "score_components": score_components,
        }

    def generate_with_comparison(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Параллельная (для прототипа — последовательная) генерация через две модели
        с выбором лучшего ответа по интегральному Score.
        """
        # Выбираем две модели
        m1 = self.models["session_default"]
        m2 = self.models["session_fast"]

        r1 = self.generate(m1.id, request)
        r2 = self.generate(m2.id, request)

        best = r1 if r1["score"] >= r2["score"] else r2
        return {
            "best": best,
            "alternatives": [r1, r2],
        }

    def evaluate_response(
        self,
        text: str,
        latency: float,
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Оценка ответа по компонентам Q, L, C, S, как описано в НИР.

        Здесь используется простая эвристика:
            - Q: длина и "структурированность" текста
            - L: нормированная латентность
            - C: приблизительная стоимость (берём константу)
            - S: педагогическая/моральная уместность (пока 0.8 по умолчанию)
        """
        # Quality (Q): чем длиннее осмысленный текст, тем выше
        length = len(text or "")
        q = max(0.0, min(1.0, length / 800.0))

        # Latency (L): меньше — лучше; нормируем так, чтобы <=2с ~ 1.0
        l = max(0.0, min(1.0, 1.0 - (latency / 2.0)))

        # Cost (C): на уровне прототипа пусть будет константой
        c = 0.5

        # Pedagogical suitability / Safety (S):
        # можно привязать к эмоциональной безопасности из контекста
        esi = float(context.get("emotional_safety_index", 0.7))
        s = max(0.0, min(1.0, esi))

        # Интегральный Score:
        #   Score = 0.4*Q + 0.2*L + 0.2*(1-C) + 0.2*S
        score = 0.4 * q + 0.2 * l + 0.2 * (1.0 - c) + 0.2 * s
        components = {"Q": q, "L": l, "C": c, "S": s}
        return float(score), components

    # ------------------------------------------------------------------
    # OpenAI / ChatGPT integration
    # ------------------------------------------------------------------

    def _call_openai_chat(
        self,
        model_name: str,
        messages: Any,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, float, Optional[Dict[str, Any]]]:
        """
        Вызов ChatGPT / OpenAI API.

        Ключ берётся из config.yaml (openai.api_key) или переменной окружения OPENAI_API_KEY.

        Если ключа нет или запрос завершается с ошибкой (401, сети и т.п.),
        метод возвращает пустой текст, чтобы оркестратор смог сделать fallback.
        """
        api_key = self._get_openai_api_key()
        if not api_key:
            return "", 0.0, None

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
        url = base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }

        started = time.time()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=(5, 30))
            latency = time.time() - started
            if resp.status_code != 200:
                # Логируем ошибку, но не падаем
                self._repo.log_event(
                    "llm_usage",
                    {
                        "provider": "openai",
                        "model_id": model_name,
                        "status_code": resp.status_code,
                        "error": resp.text[:500],
                    },
                )
                return "", latency, None
            result = resp.json()
            text = ""
            try:
                text = result["choices"][0]["message"]["content"]
            except Exception:
                text = ""
            return text, latency, result
        except Exception as e:
            latency = time.time() - started
            self._repo.log_event(
                "llm_usage",
                {
                    "provider": "openai",
                    "model_id": model_name,
                    "error": str(e),
                },
            )
            return "", latency, None

    # ------------------------------------------------------------------
    # Алиасы в стиле, использованном в НИР (camelCase)
    # ------------------------------------------------------------------

    def selectBestModel(self, context: Dict) -> ModelDescriptor:
        """Alias для совместимости с описанием в НИР."""
        return self.select_best_model(context)

    def generateWithComparison(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Alias для совместимости с описанием в НИР."""
        return self.generate_with_comparison(request)

    def evaluateResponse(
        self,
        text: str,
        latency: float,
        context: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """Alias для совместимости с описанием в НИР."""
        return self.evaluate_response(text, latency, context)

