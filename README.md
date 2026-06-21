# Hermes — the smallest slice

A local repo-memory + Q&A CLI built on **Nous Hermes 3** (open-weight,
self-hosted via Ollama). Two commands:

```bash
hermes ingest .                        # build a vector index from `git log -p`
hermes ask "why did we drop kafka?"    # agentic Q&A using a search_memory tool
```

This is the smallest slice of a larger experiment described in
[`concept.txt`](./concept.txt) — an ambient, always-on developer daemon
with a multi-agent runtime over a persistent memory layer. The slice
shipped here is the *memory + on-demand Q&A* path. No daemon, no file
watcher, no extra agents.

## Recognition

This project was written up for the DEV Community —
[Building an ambient developer daemon with Nous Hermes](https://dev.to/piwe/building-an-ambient-developer-daemon-with-nous-hermes-1667)
— which earned a trophy:

<img src="./HermesTrophy.png" alt="Hermes — DEV Community trophy" width="240" />

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** running on the same machine (or reachable
  via `OLLAMA_HOST`).
- A Hermes 3 model and an embedding model pulled into Ollama:

  ```bash
  ollama pull hermes3:8b           # default; ~4.7 GB
  ollama pull nomic-embed-text     # 768-dim embeddings; ~270 MB
  ```

  If your GPU has the room, `ollama pull hermes3:70b` is a meaningful
  quality bump and is what the agent loop is most enjoyable on.

## Install

```bash
git clone <this-repo> hermes
cd hermes
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Use

Point it at any git repo (defaults to the current directory):

```bash
hermes ingest .
hermes ask "what changed about retry handling in the last six months?"
hermes ask "who has touched the rate limiter recently and why?"
```

Re-running `ingest` wipes and rebuilds the store; for v0 there is no
incremental indexing. `--max-commits` (default 500) caps how far back it
reads.

## Configuration

All optional. Sensible defaults work for a single-user workstation.

| Env var | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is listening. |
| `HERMES_MODEL` | `hermes3:8b` | Default chat model. CLI `--model` overrides. |
| `HERMES_EMBED_MODEL` | `nomic-embed-text` | Embedding model. Must match the dim of the existing store. |

The store lives at `~/.hermes/store/` by default (override with `--store`).

## What's happening under the hood

1. **`ingest`** runs `git log -p`, parses commits into per-message and
   per-file-diff chunks, embeds each chunk via Ollama, and writes them to
   a LanceDB table on disk.
2. **`ask`** sends your question to Hermes 3 with a single tool —
   `search_memory(query, k)` — described in the standard Nous
   `<tools></tools>` system prompt. Hermes decides when to call the tool
   (usually once or twice), and the loop feeds results back as
   `<tool_response>` messages until the model emits a final answer.

That's the whole agent loop. About 80 lines across `hermes/llm.py` and
`hermes/agent.py`. Read those two files first if you want to understand
the design.

## WSL2 perf gotcha

If you're developing in WSL2 with the project living under `/mnt/c/...`,
keep the LanceDB store on the WSL-native filesystem (the default
`~/.hermes/store/` already is). Writing many small embedding rows through
DrvFs (`/mnt/c`) is *much* slower than the equivalent on `~/`. The source
code can stay on the Windows side; only the store needs to be elsewhere.

## Troubleshooting

- **`ConnectError` to Ollama** — check `ollama list` works. If Ollama is on
  another machine, `export OLLAMA_HOST=http://that-host:11434`.
- **`model "hermes3:8b" not found`** — `ollama pull hermes3:8b`.
- **Embedding step is slow** — that's expected on first ingest; embeddings
  are computed sequentially. Throughput improves dramatically with a GPU.
- **Agent answers without citing commits** — bump to `--model hermes3:70b`;
  the 8B model occasionally skips the search step on easy questions.

## What's deliberately not in v0

- File watcher and the broader daemon
- The other agents (test runner, commit helper, doc keeper, standup
  composer, router)
- Slack / Linear / GitHub adapters
- Incremental indexing
- Multi-repo memory
- Any kind of UI beyond the CLI

The shape of all of those is in [`concept.txt`](./concept.txt). The point
of this repo is to make the memory + Q&A loop *real* — small enough to
read in one sitting, big enough to actually answer questions about your
own codebase.
