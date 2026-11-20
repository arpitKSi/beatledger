"""Model transport and attempt telemetry for OpenAI-compatible endpoints."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


@dataclass(frozen=True)
class Attempt:
    status: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    http_status: int | None = None


@dataclass(frozen=True)
class ProviderReply:
    content: str | None
    attempts: tuple[Attempt, ...]
    error: str | None = None
    stop_run: bool = False


class JudgeProvider(Protocol):
    def complete(self, messages: list[dict[str, str]], schema: dict) -> ProviderReply: ...


def _loopback(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _number(name: str, value: str, *, positive: bool) -> float:
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number.") from None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'greater than zero' if positive else 'nonnegative'}.")
    return result


@dataclass(frozen=True)
class JudgeSettings:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen2.5:1.5b"
    api_key: str = field(default="", repr=False)
    response_format: str = "schema"
    timeout_seconds: float = 45
    max_output_tokens: int = 500
    temperature: float | None = 0
    allow_remote: bool = False
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    use_max_completion_tokens: bool = False

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        try:
            url.port
        except ValueError:
            raise ValueError("BEATLEDGER_BASE_URL has an invalid port.") from None
        if (
            url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
            or url.query or url.fragment
        ):
            raise ValueError("BEATLEDGER_BASE_URL must be an HTTP(S) API base URL without credentials or a query.")
        if not self.local and (url.scheme != "https" or not self.allow_remote):
            raise ValueError("A remote judge needs an HTTPS URL and explicit --allow-remote-judge opt-in.")
        if not self.model.strip():
            raise ValueError("BEATLEDGER_MODEL must name a model.")
        if self.response_format not in {"schema", "json", "prompt"}:
            raise ValueError("BEATLEDGER_RESPONSE_FORMAT must be schema, json or prompt.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or self.max_output_tokens < 1:
            raise ValueError("Judge timeout and output token limit must be positive.")
        if self.temperature is not None and (
            not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2
        ):
            raise ValueError("Judge temperature must be between 0 and 2.")
        rates = (self.input_usd_per_million, self.output_usd_per_million)
        if (rates[0] is None) != (rates[1] is None):
            raise ValueError("Configure both input and output prices, or leave both unset.")
        if any(rate is not None and (not math.isfinite(rate) or rate < 0) for rate in rates):
            raise ValueError("Token prices must be finite and nonnegative.")

    @property
    def local(self) -> bool:
        return _loopback(urlsplit(self.base_url).hostname or "")

    @classmethod
    def from_environment(cls, *, allow_remote: bool = False) -> JudgeSettings:
        env = os.environ
        input_price = env.get("BEATLEDGER_INPUT_USD_PER_MILLION")
        output_price = env.get("BEATLEDGER_OUTPUT_USD_PER_MILLION")
        try:
            max_tokens = int(env.get("BEATLEDGER_MAX_OUTPUT_TOKENS", "500"))
        except ValueError:
            raise ValueError("BEATLEDGER_MAX_OUTPUT_TOKENS must be a positive integer.") from None
        return cls(
            base_url=env.get("BEATLEDGER_BASE_URL", "http://127.0.0.1:11434/v1"),
            model=env.get("BEATLEDGER_MODEL", "qwen2.5:1.5b"),
            api_key=env.get("BEATLEDGER_API_KEY", ""),
            response_format=env.get("BEATLEDGER_RESPONSE_FORMAT", "schema"),
            timeout_seconds=_number("BEATLEDGER_TIMEOUT_SECONDS", env.get("BEATLEDGER_TIMEOUT_SECONDS", "45"), positive=True),
            max_output_tokens=max_tokens,
            temperature=_number("BEATLEDGER_TEMPERATURE", env.get("BEATLEDGER_TEMPERATURE", "0"), positive=False),
            allow_remote=allow_remote,
            input_usd_per_million=_number("Input price", input_price, positive=False) if input_price is not None else None,
            output_usd_per_million=_number("Output price", output_price, positive=False) if output_price is not None else None,
        )


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _retry_delay(value: str | None) -> float:
    if not value:
        return 1.0
    try:
        delay = float(value)
        if not math.isfinite(delay):
            return 1.0
    except ValueError:
        try:
            retry_time = parsedate_to_datetime(value)
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)
            delay = (retry_time - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 1.0
    return max(0.0, min(5.0, delay))


def _usage(usage: object, name: str) -> int | None:
    value = usage.get(name) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


class ChatCompletionsProvider:
    def __init__(self, settings: JudgeSettings):
        self.settings = settings
        self._opener = build_opener(_NoRedirect(), ProxyHandler({})) if settings.local else build_opener(_NoRedirect())

    def complete(self, messages: list[dict[str, str]], schema: dict) -> ProviderReply:
        settings = self.settings
        payload: dict = {
            "model": settings.model,
            "messages": messages,
            "stream": False,
        }
        token_limit = "max_completion_tokens" if settings.use_max_completion_tokens else "max_tokens"
        payload[token_limit] = settings.max_output_tokens
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        if settings.response_format == "schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "beatledger_judgment", "strict": True, "schema": schema},
            }
        elif settings.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        request = Request(
            f"{settings.base_url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )

        attempts = []
        for attempt_number in (1, 2):
            started = time.perf_counter()
            delay = 1.0
            try:
                with self._opener.open(request, timeout=settings.timeout_seconds) as response:
                    data = response.read(2_000_001)
            except HTTPError as exc:
                error = f"http_{exc.code}"
                retry = exc.code in {408, 429} or 500 <= exc.code <= 599
                delay = _retry_delay(exc.headers.get("Retry-After") if exc.headers is not None else None)
                attempts.append(Attempt(error, time.perf_counter() - started, http_status=exc.code))
                exc.close()
            except (OSError, HTTPException) as exc:
                reason = exc.reason if isinstance(exc, URLError) else exc
                error = "timeout" if isinstance(reason, TimeoutError) else "connection_error"
                retry = True
                attempts.append(Attempt(error, time.perf_counter() - started))
            else:
                content, error, input_tokens, output_tokens = self._decode(data)
                attempts.append(Attempt(
                    error or "response", time.perf_counter() - started, input_tokens, output_tokens, 200,
                ))
                return ProviderReply(content, tuple(attempts), error)

            if retry and attempt_number == 1:
                time.sleep(delay)
                continue
            return ProviderReply(None, tuple(attempts), error, stop_run=True)
        raise AssertionError("Judge transport exceeded its retry limit.")

    @staticmethod
    def _decode(data: bytes) -> tuple[str | None, str | None, int | None, int | None]:
        if len(data) > 2_000_000:
            return None, "provider_reply_too_large", None, None
        try:
            body = json.loads(data)
            usage = body.get("usage")
        except (ValueError, AttributeError):
            return None, "invalid_provider_reply", None, None
        tokens = (_usage(usage, "prompt_tokens"), _usage(usage, "completion_tokens"))
        try:
            first = body["choices"][0]
            message = first["message"]
            if first.get("finish_reason") == "length":
                return None, "output_truncated", *tokens
            if first.get("finish_reason") == "content_filter" or message.get("refusal"):
                return None, "provider_refusal", *tokens
            if isinstance(message.get("content"), str) and message["content"].strip():
                return message["content"], None, *tokens
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        return None, "invalid_provider_reply", *tokens
