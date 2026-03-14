# Local AI Project Guide

> The full operating manual for `local-ai`: architecture, runtime behavior, configuration, and troubleshooting in one place.

## At A Glance

| Area | Current Value |
| --- | --- |
| Default model | `Qwen/Qwen2.5-7B-Instruct` |
| UI | `http://localhost:8501` |
| vLLM API | `http://localhost:8000/v1` |
| Context window | `4096` |
| Stack style | Docker + Streamlit + vLLM + Chroma |
| Primary runtime data | `docs/`, `chroma_db/`, `chat_history/`, `vllm/cache/` |

## What's In This Guide

- project summary
- architecture and request flow
- repo layout and key files
- runtime configuration
- model and token-budget behavior
- startup and restart commands
- RAG and persistence behavior
- troubleshooting and recovery steps

## Quick Navigation

- [Project Summary](#project-summary)
- [High-Level Architecture](#high-level-architecture)
- [Main Components](#main-components)
- [Repository Layout](#repository-layout)
- [Request Flow](#request-flow)
- [Current Runtime Configuration](#current-runtime-configuration)
- [Environment Variables](#environment-variables)
- [How To Start The Stack](#how-to-start-the-stack)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## Project Summary

`local-ai` is a self-hosted local assistant stack built around:

- a Streamlit chat UI
- a vLLM OpenAI-compatible inference server
- optional retrieval-augmented generation (RAG) using Chroma
- local disk persistence for uploaded documents and chat sessions

### Current default model

- `Qwen/Qwen2.5-7B-Instruct`

This project is designed to run on a single machine with Docker and an NVIDIA GPU.

> Why this model: it gives this machine much better VRAM headroom and a far more practical context budget than the earlier 14B setup.

## High-Level Architecture

```text
Browser
  -> Streamlit app (chat-ui/app.py)
      -> vLLM HTTP API (/v1/chat/completions, /v1/models)
          -> Qwen model running on GPU

Optional RAG flow:
uploaded file
  -> saved to docs/
  -> parsed by chat-ui/app.py
  -> chunked + embedded by chat-ui/rag.py
  -> stored in chroma_db/
  -> retrieved during chat
  -> appended to the system prompt
```

## Main Components

| Component | Main File | Purpose |
| --- | --- | --- |
| Streamlit UI | `chat-ui/app.py` | Chat interface, session state, uploads, prompt building |
| vLLM server | `chat-ui/docker-compose.yml` | Model serving, OpenAI-compatible API, GPU inference |
| RAG layer | `chat-ui/rag.py` | Chunking, embeddings, Chroma indexing and retrieval |
| Runtime storage | `docs/`, `chroma_db/`, `chat_history/`, `vllm/cache/` | Persisted documents, vectors, sessions, and model cache |

### 1. Streamlit Chat UI

File:

- `chat-ui/app.py`

Responsibilities:

- renders the browser UI
- shows model/server status
- manages saved chat sessions
- uploads and manages documents
- builds prompts and request payloads
- streams model responses from vLLM
- trims history and output token requests to fit the model context budget

### 2. vLLM Model Server

File:

- `chat-ui/docker-compose.yml`

Responsibilities:

- serves the active LLM over an OpenAI-compatible API
- loads the current model into GPU memory
- exposes routes like `/v1/models` and `/v1/chat/completions`
- performs generation for the UI

### 3. RAG Layer

File:

- `chat-ui/rag.py`

Responsibilities:

- extracts chunks from uploaded documents
- creates embeddings
- stores vectors in Chroma
- retrieves relevant chunks for a user query
- formats retrieval context for prompt injection

### 4. Persistent Runtime Storage

Directories:

- `docs/`
- `chroma_db/`
- `chat_history/`
- `vllm/cache/`

Responsibilities:

- `docs/`: uploaded source documents
- `chroma_db/`: persisted vector store data
- `chat_history/`: saved conversations
- `vllm/cache/`: Hugging Face and vLLM model cache data

## Repository Layout

```text
local-ai/
|-- chat-ui/
|   |-- app.py
|   |-- rag.py
|   |-- docker-compose.yml
|   |-- dockerfile
|   |-- requirements.txt
|   `-- tests/
|       `-- test_rag.py
|-- chroma_db/
|-- chat_history/
|-- docs/
|-- models/
|-- scripts/
|   `-- rewrite_rag.py
|-- vllm/
|   `-- cache/
|-- LICENSE
|-- pytest.ini
|-- README.md
`-- docs/PROJECT_GUIDE.md
```

## Request Flow

### Normal Chat

1. The user sends a prompt in the Streamlit UI.
2. The app adds recent chat history.
3. The app applies token budgeting so the request fits the current context window.
4. The app sends a streaming request to `vLLM_API_BASE + /chat/completions`.
5. vLLM generates tokens on the GPU.
6. The UI streams the partial output to the browser.
7. The final reply is saved to `chat_history/`.

### Chat With RAG

1. The user uploads `txt`, `md`, or `pdf` files.
2. The app saves them to `docs/`.
3. The text is extracted and chunked.
4. Embeddings are generated and stored in Chroma.
5. On chat, the app searches the vector store for relevant chunks.
6. Retrieved context is inserted into the system prompt.
7. The model responds with that context available.

## Current Runtime Configuration

The active compose stack lives in:

- `chat-ui/docker-compose.yml`

### Runtime Snapshot

| Item | Value |
| --- | --- |
| Services | `vllm`, `chat-ui` |
| UI port | `8501` |
| API port | `8000` |
| Default model | `Qwen/Qwen2.5-7B-Instruct` |
| Compose healthcheck | `python3` request to `/v1/models` |

### Current vLLM Settings

| Flag | Value |
| --- | --- |
| `--dtype` | `bfloat16` |
| `--gpu-memory-utilization` | `0.90` |
| `--max-model-len` | `4096` |
| `--max-num-batched-tokens` | `2048` |
| `--max-num-seqs` | `4` |

These settings are intentionally more conservative than the earlier 14B setup and provide much better memory headroom on this machine.

## Environment Variables

### `chat-ui` container

| Variable | Current Value | Purpose |
| --- | --- | --- |
| `VLLM_API_BASE` | `http://vllm:8000/v1` | Base URL for model requests |
| `MODEL_NAME` | `Qwen/Qwen2.5-7B-Instruct` | Default selected model in the UI |
| `VLLM_MAX_MODEL_LEN` | `4096` | UI-side token budgeting and trimming |
| `DOCS_DIR` | `/docs` | Uploaded document storage |
| `CHROMA_DB_PATH` | `/chroma_db` | Vector store location |
| `CHAT_HISTORY_DIR` | `/chat_history` | Chat session storage |
| `HUGGING_FACE_HUB_TOKEN` / `HF_TOKEN` | user-provided | Hugging Face downloads |

### `vllm` container

| Variable | Purpose |
| --- | --- |
| `HUGGING_FACE_HUB_TOKEN` | model access and downloads |

## Windows-Specific Mounted Paths

> This compose file is currently tailored to this Windows machine. If the repo moves, update these paths first.

The compose file currently assumes this machine layout:

- `G:/local-ai/vllm/cache:/root/.cache/huggingface`
- `G:/local-ai/docs:/docs`
- `G:/local-ai/chroma_db:/chroma_db`
- `G:/local-ai/chat_history:/chat_history`

If you move the repo or run it on another machine, these are the first paths to update.

## How To Start The Stack

From `g:\local-ai\chat-ui`:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

## How To Restart The Stack

From `g:\local-ai\chat-ui`:

```bash
docker compose up -d
```

To restart only the model server:

```bash
docker compose up -d vllm
```

To restart only the UI:

```bash
docker compose up -d chat-ui
```

## How The UI Prevents Token Errors

The app now includes token-budget logic in `chat-ui/app.py`.

It does three important things:

- caps the output token slider to a safer range
- trims older chat history when the request would exceed the model context window
- uses `VLLM_MAX_MODEL_LEN` so the UI stays aligned with the current vLLM configuration

This was added after the earlier `Qwen/Qwen3-14B` setup caused request-size and context-window failures.

## Model Choice Notes

| Model | Outcome On This Machine |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | Stable, healthy startup, useful context headroom |
| `Qwen/Qwen3-14B` | Too tight on VRAM for comfortable vLLM KV cache and chat usage |

### Why the 7B model is the better fit

- much lower VRAM usage than the 14B model
- much larger usable KV cache
- fewer token-limit failures
- better room for multi-turn chat and RAG
- faster and more stable startup

### Why the 14B model caused trouble

- it fit poorly within available GPU headroom on this system
- after weights loaded, KV cache space was too small
- the system had to run with a severely reduced context window
- chat requests frequently ran into token pressure

## Chat Session Storage

Chat sessions are stored as JSON files in:

- `chat_history/`

Each session includes:

- title
- created timestamp
- saved timestamp
- message list

The app automatically:

- creates a new session if none exist
- updates the title from user content
- saves messages after responses complete

## Document Processing And RAG

Supported upload types:

- `.txt`
- `.md`
- `.pdf`

Processing flow:

1. file is uploaded in the sidebar
2. file is saved to `docs/`
3. text is extracted
4. chunks are created
5. embeddings are generated
6. chunks are stored in Chroma

RAG controls in the UI:

- enable or disable document search
- show or hide retrieved context
- preview uploaded files
- search inside saved documents
- re-index a selected document
- remove a selected document
- clear the indexed knowledge base

## Configuration Checklist

If you want to change the model or machine layout, update these first:

1. `chat-ui/docker-compose.yml`
2. `chat-ui/app.py` defaults only if you want code-level fallback changes
3. `README.md` or this guide if you want docs to match

When changing models, review:

- `--model`
- `MODEL_NAME`
- `VLLM_MAX_MODEL_LEN`
- `--gpu-memory-utilization`
- `--max-model-len`
- `--max-num-batched-tokens`
- `--max-num-seqs`

## Troubleshooting

> Start with `docker logs --tail 200 vllm-server` and `docker inspect vllm-server --format '{{json .State.Health}}'` for almost every model-serving issue.

### 1. The chat UI loads, but the model is unavailable

Symptoms:

- sidebar shows model API unavailable
- chat requests fail immediately

Checks:

- confirm `vllm-server` is running
- confirm `http://localhost:8000/v1/models` responds
- inspect container logs

Useful commands:

```bash
docker ps -a
docker logs --tail 200 vllm-server
docker inspect vllm-server --format '{{json .State.Health}}'
```

### 2. vLLM crashes during startup with cache or memory errors

Common error patterns:

- `No available memory for the cache blocks`
- `Free memory on device ... is less than desired GPU memory utilization`
- `estimated maximum model length is ...`

What it means:

- the model weights fit badly enough that there is not enough remaining VRAM for KV cache or warmup

Fixes:

- lower `--max-model-len`
- lower `--max-num-batched-tokens`
- lower `--max-num-seqs`
- lower `--gpu-memory-utilization` if startup target is above real free VRAM
- switch to a smaller model

### 3. The model loads, but chat shows token-limit errors

Symptoms:

- token error appears in the chat
- request is rejected even though vLLM is healthy

What it usually means:

- the request payload is too large for the model context window

Fixes:

- reduce `max_tokens`
- reduce retrieval context size
- reduce number of history messages
- ensure `VLLM_MAX_MODEL_LEN` matches the actual server setting

Current status:

- the app now trims requests automatically to reduce this failure mode

### 4. Container healthcheck fails even though the server should work

Previous issue:

- healthcheck used `python`, which was not available in the image

Current fix:

- the compose file now uses `python3`

If health keeps failing, verify:

- the server has finished model loading
- `/v1/models` responds inside the container

### 5. Startup takes a long time after reboot

This can be normal.

Why:

- large model weights must be loaded
- vLLM performs compile and warmup work
- CUDA graph capture takes additional time

The first startup after a change or reboot is usually slower than later runs.

### 6. GPU memory seems unexpectedly full

Check:

```bash
nvidia-smi
```

Look for:

- other GPU-heavy apps
- desktop or background apps consuming VRAM
- a model that is too large for comfortable headroom

If free VRAM is low:

- close GPU-heavy apps
- use a smaller model
- reduce vLLM limits

### 7. Uploaded PDFs fail to index

Possible causes:

- encrypted PDF
- no extractable text
- malformed file

The app already handles these with user-facing errors, but the fix is usually to:

- use an unencrypted PDF
- export the file again
- convert it to text or markdown first

### 8. RAG results feel weak or repetitive

Likely causes:

- source documents are too noisy
- chunks are too similar
- retrieved context is too large or too generic

Things to try:

- upload cleaner documents
- re-index the document
- remove outdated docs from the knowledge base
- reduce prompt clutter

### 9. The UI works, but changes do not show up

Check:

- whether the `chat-ui` container was restarted
- whether Streamlit is reading the current bind-mounted files

Useful command:

```bash
docker compose up -d chat-ui
```

## Verification Commands

### Fastest health workflow

1. Check containers.
2. Check vLLM health.
3. Check vLLM logs.
4. Check GPU memory if startup is failing.

These are the fastest commands to check overall health.

### Container status

```bash
docker ps -a
```

### vLLM logs

```bash
docker logs --tail 200 vllm-server
```

### vLLM health

```bash
docker inspect vllm-server --format '{{json .State.Health}}'
```

### GPU memory

```bash
nvidia-smi
```

## Recommended Next Improvements

- add a small GitHub Action once the test flow is stable
- move machine-specific paths to a `.env` file or compose overrides
- add a lightweight smoke test for `/v1/models`
- add a startup check in the UI that explains when the model is still warming up
- consider documenting model profiles such as 7B vs 14B in a separate comparison page

## Quick Reference

### Current stable local setup

| Setting | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-7B-Instruct` |
| UI | `http://localhost:8501` |
| API | `http://localhost:8000/v1` |
| vLLM context window | `4096` |
| output trimming | enabled in app |
| RAG persistence | enabled |
| chat persistence | enabled |

### Most important files

- `chat-ui/app.py`
- `chat-ui/rag.py`
- `chat-ui/docker-compose.yml`
- `README.md`
- `docs/PROJECT_GUIDE.md`
