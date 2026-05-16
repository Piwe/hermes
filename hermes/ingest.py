from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from . import store as store_mod
from .llm import embed

_COMMIT_SEP = "===COMMIT==="
_HEADER_END = "===END_HEADER==="


def _git_log(repo_path: Path, max_commits: int) -> str:
    fmt = f"{_COMMIT_SEP}%n%H%n%an%n%aI%n%s%n%b%n{_HEADER_END}"
    cmd = [
        "git", "-C", str(repo_path), "log",
        f"--pretty=format:{fmt}",
        "-p", "--no-color", "--no-merges",
        f"-{max_commits}",
    ]
    return subprocess.check_output(cmd, text=True, errors="replace")


def parse_git_log(text: str) -> list[dict]:
    commits = []
    for block in text.split(_COMMIT_SEP + "\n")[1:]:
        try:
            header, body = block.split(_HEADER_END + "\n", 1)
        except ValueError:
            continue
        lines = header.split("\n")
        if len(lines) < 4:
            continue
        sha, author, date, subject = lines[0], lines[1], lines[2], lines[3]
        msg_body = "\n".join(lines[4:]).strip()
        commits.append({
            "sha": sha,
            "author": author,
            "date": date,
            "subject": subject,
            "message": (subject + ("\n\n" + msg_body if msg_body else "")).strip(),
            "diff": body.strip(),
        })
    return commits


def split_diff_by_file(diff: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    cur_path: str | None = None
    cur_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if cur_path is not None:
                blocks.append((cur_path, "\n".join(cur_lines)))
            cur_lines = [line]
            parts = line.split()
            cur_path = parts[3].removeprefix("b/") if len(parts) >= 4 else "?"
        else:
            cur_lines.append(line)
    if cur_path is not None:
        blocks.append((cur_path, "\n".join(cur_lines)))
    return blocks


def chunks_for_commit(commit: dict, max_diff_chars: int = 8000) -> Iterable[dict]:
    base = {
        "commit_hash": commit["sha"],
        "author": commit["author"],
        "date": commit["date"],
    }
    yield {
        **base,
        "id": f"{commit['sha']}::msg",
        "source": "commit_message",
        "file_path": "",
        "content": commit["message"],
    }
    if not commit["diff"]:
        return
    for path, block in split_diff_by_file(commit["diff"]):
        yield {
            **base,
            "id": f"{commit['sha']}::{path}",
            "source": "diff",
            "file_path": path,
            "content": block[:max_diff_chars],
        }


def ingest(
    repo_path: Path,
    store_path: Path,
    max_commits: int = 500,
    embed_dim: int = 768,
    progress=None,
) -> int:
    """Wipe the store and rebuild from `git log -p`. Returns commit count."""
    raw = _git_log(repo_path, max_commits)
    commits = parse_git_log(raw)
    table = store_mod.reset(store_path, dim=embed_dim)

    def row_iter():
        for i, commit in enumerate(commits):
            for chunk in chunks_for_commit(commit):
                vec = embed(chunk["content"])
                yield {**chunk, "vector": vec}
            if progress is not None:
                progress(i + 1, len(commits), commit["sha"][:8])

    store_mod.add_rows(table, row_iter())
    return len(commits)
