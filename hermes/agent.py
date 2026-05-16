from __future__ import annotations

from pathlib import Path
from typing import Any

from . import store as store_mod
from .llm import (
    DEFAULT_MODEL,
    chat,
    embed,
    format_tool_response,
    format_tools_system_prompt,
    parse_tool_calls,
)

SEARCH_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": (
            "Search the repository memory (commit messages and diffs) for chunks "
            "relevant to a query. Returns the top matches with their source, "
            "commit hash, author, date, and content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Short search phrase or question.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many matches to return (default 5, max 12).",
                },
            },
            "required": ["query"],
        },
    },
}

SYSTEM_INSTRUCTIONS = (
    "You are Hermes, a local assistant that answers questions about the user's "
    "codebase using the repository memory tool. Always call search_memory at "
    "least once before answering. Cite specific commits by short hash and date "
    "when relevant. If the memory does not contain the answer, say so honestly "
    "rather than guessing."
)


def _search(table, query: str, k: int) -> list[dict]:
    k = max(1, min(int(k), 12))
    vec = embed(query)
    rows = store_mod.search(table, vec, k=k)
    return [
        {
            "source": r.get("source"),
            "commit": (r.get("commit_hash") or "")[:8],
            "author": r.get("author", ""),
            "date": r.get("date", ""),
            "file": r.get("file_path", ""),
            "content": (r.get("content") or "")[:1500],
        }
        for r in rows
    ]


def ask(
    question: str,
    store_path: Path,
    model: str = DEFAULT_MODEL,
    max_iters: int = 4,
) -> str:
    table = store_mod.open_or_create(store_path)
    system = format_tools_system_prompt([SEARCH_MEMORY_TOOL]) + "\n\n" + SYSTEM_INSTRUCTIONS
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    for _ in range(max_iters):
        response = chat(messages, model=model)
        calls = parse_tool_calls(response)
        if not calls:
            return response.strip()
        messages.append({"role": "assistant", "content": response})
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {}) or {}
            if name == "search_memory":
                results = _search(table, args.get("query", question), args.get("k", 5))
                tool_msg = format_tool_response(name, results)
            else:
                tool_msg = format_tool_response(name, {"error": f"unknown tool: {name}"})
            messages.append({"role": "tool", "content": tool_msg})
    return "(reached max iterations without a final answer)"
