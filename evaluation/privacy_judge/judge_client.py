#!/usr/bin/env python3
"""OpenAI-compatible multimodal judge client for privacy trajectory evaluation."""

from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import httpx


def _openai_api_config() -> tuple[str, str]:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    missing = []
    if not base_url:
        missing.append("OPENAI_BASE_URL")
    if not api_key:
        missing.append("OPENAI_API_KEY")
    if missing:
        raise RuntimeError(f"Missing OpenAI API environment variable(s): {', '.join(missing)}")

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url, api_key


class JudgeClient:
    """Thin wrapper around OpenAI API multimodal judge APIs."""

    SUPPORTED_APIS = {"responses", "chat-completions"}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-5.4",
        timeout: float = 60.0,
        max_retries: int = 10,
        temperature: float = 0.0,
        api: str = "responses",
    ) -> None:
        env_base_url, env_api_key = _openai_api_config()
        api = api.strip().lower()
        if api not in self.SUPPORTED_APIS:
            raise ValueError(f"Unsupported judge API {api!r}; choose one of {sorted(self.SUPPORTED_APIS)}")
        self._model = model
        self._api_key = api_key or env_api_key
        self._base_url = (base_url or env_base_url).rstrip("/")
        if not self._base_url.endswith("/v1"):
            self._base_url = f"{self._base_url}/v1"
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._temperature = temperature
        self._api = api

    @staticmethod
    def _retry_sleep_seconds(attempt: int) -> float:
        base_delay = min(0.5 * (2 ** max(0, attempt - 1)), 8.0)
        return base_delay + random.uniform(0.0, base_delay * 0.25)

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        suffix = image_path.suffix.lower()
        if suffix == ".png":
            mime_type = "image/png"
        elif suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        else:
            raise ValueError(f"Unsupported image suffix: {image_path.suffix}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _build_content(self, prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_path)},
                }
            )
        return content

    def _build_responses_content(self, prompt: str, image_paths: list[Path]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_to_data_url(image_path),
                }
            )
        return content

    @staticmethod
    def _extract_chat_message_text(message_content: Any) -> str:
        if isinstance(message_content, str):
            return message_content
        if not isinstance(message_content, list):
            return ""

        chunks: list[str] = []
        for item in message_content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    @staticmethod
    def _extract_response_text(response_payload: dict[str, Any]) -> str:
        chunks: list[str] = []
        for item in response_payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text") or content.get("output_text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)

    def _judge_responses(self, prompt: str, image_paths: list[Path]) -> tuple[str, dict[str, Any]]:
        content = self._build_responses_content(prompt, image_paths)
        payload = {
            "model": self._model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": content,
                }
            ],
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        text_chunks: list[str] = []
        final_response: dict[str, Any] | None = None
        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/responses",
                headers=headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    raw_event = line[len("data:"):].strip()
                    if not raw_event or raw_event == "[DONE]":
                        continue
                    event = json.loads(raw_event)
                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            text_chunks.append(delta)
                    elif event_type == "response.output_text.done":
                        text = event.get("text")
                        if isinstance(text, str) and not text_chunks:
                            text_chunks.append(text)
                    elif event_type == "response.completed":
                        final_response = event.get("response") or {}
                    elif event_type == "error" or event.get("error"):
                        raise RuntimeError(json.dumps(event.get("error") or event, ensure_ascii=False))

        raw_text = "".join(text_chunks)
        if not raw_text and final_response:
            raw_text = self._extract_response_text(final_response)
        meta = {
            "api": "responses",
            "finish_reason": final_response.get("status") if final_response else None,
            "usage": final_response.get("usage") if final_response else None,
        }
        return raw_text, meta

    def _judge_chat_completions(self, prompt: str, image_paths: list[Path]) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_content(prompt, image_paths),
                }
            ],
            "temperature": self._temperature,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            response_payload = response.json()

        choices = response_payload.get("choices") or []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message") if isinstance(first_choice, dict) else {}
        raw_text = ""
        if isinstance(message, dict):
            raw_text = self._extract_chat_message_text(message.get("content"))

        meta = {
            "api": "chat-completions",
            "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
            "usage": response_payload.get("usage"),
        }
        return raw_text, meta

    def judge(self, prompt: str, image_paths: list[Path]) -> tuple[str, dict[str, Any]]:
        """Submit one multimodal judge request and return raw text plus metadata."""
        if not image_paths:
            raise ValueError("At least one image is required for multimodal judging.")

        last_error: str | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                if self._api == "chat-completions":
                    raw_text, meta = self._judge_chat_completions(prompt, image_paths)
                else:
                    raw_text, meta = self._judge_responses(prompt, image_paths)
                meta["attempt"] = attempt
                return raw_text, meta
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                if attempt < self._max_retries:
                    time.sleep(self._retry_sleep_seconds(attempt))

        raise RuntimeError(last_error or "Unknown API error")
