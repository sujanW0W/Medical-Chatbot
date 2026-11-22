"""
Selects which chat model backend the pipeline uses, via the LLM_PROVIDER
env var. Every call site in this codebase (route_tools, SynthesisAgent,
eval/run_eval.py's judge) only ever calls `.invoke(messages)` / reads
`.content` on whatever this returns - that's LangChain's own BaseChatModel
interface, already shared by ChatGoogleGenerativeAI and ChatOpenAI, so
nothing downstream needs to branch on provider.

  LLM_PROVIDER=google (default) - Google Gemini, via GOOGLE_API_KEY
  LLM_PROVIDER=openai           - any OpenAI-compatible endpoint (vLLM,
                                  OpenAI itself, etc.), via OPENAI_BASE_URL /
                                  OPENAI_API_KEY / OPENAI_MODEL

LLM_PROVIDER=openai is meant for local dev/eval (e.g. pointing at a
self-hosted vLLM server to sidestep Gemini's free-tier quota) - production
is expected to run LLM_PROVIDER=google with billing enabled, or another
billed provider added the same way later.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_chat_model() -> BaseChatModel:
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL")
        if not base_url or not model:
            raise ValueError(
                "LLM_PROVIDER=openai requires both OPENAI_BASE_URL and "
                "OPENAI_MODEL to be set (e.g. your vLLM server's /v1 URL "
                "and the model id it was launched with)."
            )
        return ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )

    if provider == "google":
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model)

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r} (expected 'google' or 'openai')"
    )
