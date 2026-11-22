"""
Shared helpers for dealing with raw chat-model output, independent of which
provider produced it (src/llm_provider.py).
"""
import re

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(content: str) -> str:
    """Remove a leading <think>...</think> block some reasoning-tuned models
    (seen so far via LLM_PROVIDER=openai against vLLM) prepend to every
    response. No-op for models that don't emit one."""
    return _THINK_BLOCK_RE.sub("", content, count=1).strip()
