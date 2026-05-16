from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("HERMES_MODEL", "hermes3:8b")
DEFAULT_EMBED_MODEL = os.getenv("HERMES_EMBED_MODEL", "nomic-embed-text")

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def chat(messages: list[dict[str, str]], model: str = DEFAULT_MODEL, **opts: Any) -> str:
    r = httpx.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": model, "messages": messages, "stream": False, **opts},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


def embed(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    r = httpx.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def format_tools_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build the Hermes function-calling system prompt.

    Hermes models are trained to look for tool schemas inside <tools></tools>
    and to emit calls inside <tool_call></tool_call>. The phrasing below
    matches Nous's reference template.
    """
    schemas = "\n".join(json.dumps(t) for t in tools)
    return (
        "You are a function calling AI model. You are provided with function "
        "signatures within <tools></tools> XML tags. You may call one or more "
        "functions to assist with the user query. Don't make assumptions about "
        "what values to plug into functions.\n\n"
        f"<tools>\n{schemas}\n</tools>\n\n"
        "For each function call return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags as follows:\n"
        '<tool_call>{"name": "<function-name>", "arguments": <args-dict>}</tool_call>'
    )


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            continue
    return calls


def format_tool_response(name: str, content: Any) -> str:
    payload = json.dumps({"name": name, "content": content}, default=str)
    return f"<tool_response>\n{payload}\n</tool_response>"
