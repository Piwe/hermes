from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import lancedb
import pyarrow as pa

TABLE_NAME = "memory"


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("commit_hash", pa.string()),
            pa.field("author", pa.string()),
            pa.field("date", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def open_or_create(path: str | Path, dim: int = 768):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(path))
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_schema(dim))


def reset(path: str | Path, dim: int = 768):
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(path))
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_schema(dim))


def add_rows(table, rows: Iterable[dict[str, Any]], batch: int = 100) -> int:
    buf: list[dict[str, Any]] = []
    n = 0
    for row in rows:
        buf.append(row)
        if len(buf) >= batch:
            table.add(buf)
            n += len(buf)
            buf = []
    if buf:
        table.add(buf)
        n += len(buf)
    return n


def search(table, query_vec: list[float], k: int = 5) -> list[dict[str, Any]]:
    return table.search(query_vec).limit(k).to_list()
