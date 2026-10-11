"""Built-in agents for real models, so you can benchmark a model with no code.

    halflife bench --model ollama:qwen2.5:7b
    halflife bench --model anthropic:claude-sonnet-5-5
    halflife bench --model openai:gpt-4.1
    halflife bench --model openrouter:meta-llama/llama-3.3-70b-instruct
    halflife bench --model vllm:my-model --base-url http://gpu-box:8000/v1

Providers and where they read their key:
    ollama       local, no key           OLLAMA_HOST (default http://localhost:11434)
    lmstudio     local, no key           http://localhost:1234/v1
    vllm         local, no key           http://localhost:8000/v1
    openai       OPENAI_API_KEY
    openrouter   OPENROUTER_API_KEY
    together     TOGETHER_API_KEY
    groq         GROQ_API_KEY
    anthropic    ANTHROPIC_API_KEY
    compatible   any OpenAI-compatible server; pass --base-url (key from HALFLIFE_API_KEY)
"""

from __future__ import annotations

import os
from typing import Optional

from ._http import ModelAPIError
from .anthropic_api import AnthropicAgent
from .openai_compat import OpenAICompatibleAgent

PROVIDERS = {
    "ollama": (None, None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "compatible": (None, "HALFLIFE_API_KEY"),
}


def from_spec(spec: str, base_url: Optional[str] = None, temperature: float = 0.0):
    """Build an agent from 'provider:model', e.g. 'ollama:qwen2.5:7b'."""
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"model must look like provider:model, e.g. ollama:qwen2.5:7b (got '{spec}')")
    name = spec
    if provider == "anthropic":
        return AnthropicAgent(model, base_url=base_url or "https://api.anthropic.com", name=name)
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider '{provider}'. Known: anthropic, {', '.join(PROVIDERS)}")
    default_url, key_var = PROVIDERS[provider]
    if provider == "ollama":
        default_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1"
        if not default_url.startswith("http"):
            default_url = "http://" + default_url
    url = base_url or default_url
    if not url:
        raise ValueError(f"provider '{provider}' needs --base-url")
    key = os.environ.get(key_var) if key_var else None
    if key_var and not key and provider not in ("compatible",):
        raise ModelAPIError(f"{key_var} is not set")
    return OpenAICompatibleAgent(model, url, api_key=key, temperature=temperature, name=name)


def preflight(agent, timeout: float = 4.0) -> Optional[str]:
    """Return a plain-English problem if the model server can't be reached, else None."""
    import urllib.error
    import urllib.request

    if isinstance(agent, AnthropicAgent):
        return None if agent.api_key else "ANTHROPIC_API_KEY is not set"
    url = agent.base_url.rstrip("/") + "/models"
    headers = {"authorization": f"Bearer {agent.api_key}"} if agent.api_key else {}
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
        return None
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return f"the server at {agent.base_url} rejected the API key ({exc.code})"
        return None
    except (urllib.error.URLError, OSError):
        host = agent.base_url.split("/v1")[0]
        if "11434" in host:
            return (
                f"can't reach Ollama at {host}. Is it installed and running? Start it with:\n"
                "    OLLAMA_CONTEXT_LENGTH=32768 ollama serve\n"
                "  and pull the model first, e.g.: ollama pull qwen2.5:7b"
            )
        return f"can't reach the model server at {host}. Is it running?"


__all__ = ["preflight", "AnthropicAgent", "OpenAICompatibleAgent", "ModelAPIError", "from_spec", "PROVIDERS"]
