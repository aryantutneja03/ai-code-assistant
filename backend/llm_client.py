"""LLM client: OpenAI when configured, else a deterministic local stub.

The app targets the OpenAI API for both QA and code generation. When no API key
is present it falls back to a local, deterministic stub so the demo always runs
offline.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings


@dataclass
class LLMResult:
    text: str
    model_used: str
    provider: str


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._openai = None
        self._gemini = None
        if settings.openai_enabled:
            try:
                from openai import OpenAI

                self._openai = OpenAI(api_key=settings.openai_api_key)
            except Exception:
                self._openai = None
        if self._openai is None and settings.gemini_enabled:
            try:
                from google import genai

                self._gemini = genai.Client(api_key=settings.gemini_api_key)
            except Exception:
                self._gemini = None

    def generate(self, system: str, user: str, mode: str) -> LLMResult:
        if self._openai is not None:
            try:
                return self._gen_openai(system, user)
            except Exception:
                pass
        if self._gemini is not None:
            try:
                return self._gen_gemini(system, user)
            except Exception:
                pass
        return self._gen_local(system, user, mode)

    def stream(self, system: str, user: str, mode: str):
        """Yield text chunks. Falls back to local token streaming offline."""
        if self._openai is not None:
            try:
                yield from self._stream_openai(system, user)
                return
            except Exception:
                pass
        if self._gemini is not None:
            try:
                yield from self._stream_gemini(system, user)
                return
            except Exception:
                pass
        result = self._gen_local(system, user, mode)
        for token in result.text.split(" "):
            yield token + " "

    # ---- OpenAI --------------------------------------------------------------
    def _gen_openai(self, system: str, user: str) -> LLMResult:
        model = self.settings.openai_chat_model
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
        )
        return LLMResult(resp.choices[0].message.content, model, "openai")

    def _stream_openai(self, system: str, user: str):
        model = self.settings.openai_chat_model
        stream = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ---- Gemini --------------------------------------------------------------
    def _gen_gemini(self, system: str, user: str) -> LLMResult:
        from google.genai import types

        model = self.settings.gemini_chat_model
        resp = self._gemini.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.1
            ),
        )
        return LLMResult(resp.text, model, "gemini")

    def _stream_gemini(self, system: str, user: str):
        from google.genai import types

        model = self.settings.gemini_chat_model
        stream = self._gemini.models.generate_content_stream(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.1
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    # ---- Local fallback ------------------------------------------------------
    def _gen_local(self, system: str, user: str, mode: str) -> LLMResult:
        context = user.split("Context from the codebase:")[-1]
        snippet = context.strip().splitlines()[:8]
        body = "\n".join(snippet)
        if mode == "codegen":
            text = (
                "```python\n"
                "# (local stub) generated from retrieved context\n"
                "def example_from_context():\n"
                "    \"\"\"Replace with real generation by setting OPENAI_API_KEY.\"\"\"\n"
                "    return 'see cited context'\n"
                "```\n"
                "Why:\n"
                "- Local stub: set OPENAI_API_KEY to get real model output.\n"
                "- Grounded on the retrieved context shown above."
            )
        else:
            text = (
                "(local stub answer — set OPENAI_API_KEY for real generation)\n\n"
                "Based on the retrieved context:\n" + body
            )
        return LLMResult(text, "local-stub", "local")
