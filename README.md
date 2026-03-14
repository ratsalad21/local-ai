# Local AI

Local AI is a self-hosted chat stack for running a local assistant with a Streamlit UI, a vLLM inference server, and optional document retrieval through Chroma.

The repo is intentionally small: one app in [`chat-ui`](./chat-ui), a Docker Compose setup for the UI and model server, and local state directories for documents, embeddings, and model cache.

## Overview

This project currently gives you:

- a browser-based local chat interface
- streaming responses from a vLLM OpenAI-compatible endpoint
- document upload for `txt`, `md`, and `pdf`
- persistent local RAG storage with Chroma
- multiple saved chat sessions
- live model selection from the available vLLM models
- sidebar controls to inspect, re-index, and remove indexed documents
- document preview and text search in the sidebar
- persistent chat history saved to local disk
- a sidebar status panel for model connectivity and local app state
- retrieved source display in the chat UI
- filtered retrieval that reduces duplicate and weak context
- a simple Docker-first setup for running everything on one machine

It is best read in two ways:

- as a GitHub project page that explains what the repo contains
- as a practical setup guide for running the current stack locally

## Current Architecture

```text
Browser
  -> Streamlit app (`chat-ui/app.py`)
  -> vLLM API (`/v1/chat/completions`)
  -> local Qwen model on GPU

Optional retrieval flow:
uploaded file
  -> saved under `docs/`
  -> parsed by the Streamlit app
  -> chunked and embedded in `chat-ui/rag.py`
  -> stored in `chroma_db/`
  -> retrieved context added to the prompt
```

## Quick Start

From [`chat-ui`](./chat-ui):

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

The current compose stack starts:

- `vllm` on port `8000`
- `chat-ui` on port `8501`

The current default model is `Qwen/Qwen3-14B`.

## Hardware Requirements

Recommended for the current default setup:

- NVIDIA GPU with at least 16 GB VRAM for comfortable local inference with `Qwen/Qwen2.5-7B-Instruct`
- 32 GB system RAM
- modern multi-core CPU
- SSD storage for the repo, model cache, and vector database
- Docker with GPU support configured for NVIDIA

Minimum practical baseline:

- NVIDIA GPU with 8 GB to 12 GB VRAM, depending on model choice and vLLM settings
- 16 GB system RAM
- enough free disk space for model weights, cache files, documents, and Chroma data

Notes:

- larger models will need substantially more VRAM than the default 7B model
- RAG and embeddings also consume CPU, RAM, and disk in addition to model inference
- if you run into memory pressure, lowering model size or vLLM memory settings is usually the first thing to adjust

## What This Repo Contains

- `chat-ui/app.py`: main Streamlit chat application
- `chat-ui/rag.py`: document chunking, embeddings, Chroma storage, and retrieval
- `chat-ui/docker-compose.yml`: two-service local stack for `vllm` and `chat-ui`
- `chat-ui/dockerfile`: container image definition for the Streamlit app
- `chat-ui/requirements.txt`: Python dependencies
- `chat-ui/tests/test_rag.py`: tests for the RAG layer
- `scripts/rewrite_rag.py`: local helper script from earlier iteration work
- `docs/`: uploaded documents stored at runtime
- `chroma_db/`: persisted vector database files
- `chat_history/`: persisted chat session storage
- `vllm/cache/`: Hugging Face model cache for the inference container
- `models/`: reserved local model directory, not currently used by the app code

## Repo Layout

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
`-- README.md
```

## How It Works Today

The Streamlit app in [`chat-ui/app.py`](./chat-ui/app.py):

- keeps chat history in Streamlit session state and saves it to disk
- sends chat requests directly to the vLLM HTTP API
- supports model selection, a custom system prompt, temperature, and max token controls
- supports multiple saved chat sessions
- includes a sidebar status panel for API reachability, indexed docs, and saved chat state
- includes sidebar tools for managing indexed documents
- includes a document explorer with preview and text search
- can optionally fetch retrieved document context before generation
- shows retrieved sources and chunk details in the UI
- stores source names alongside assistant replies when RAG is used

The RAG layer in [`chat-ui/rag.py`](./chat-ui/rag.py):

- splits documents into overlapping word chunks
- uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- stores vectors in a Chroma collection named `documents`
- can list indexed documents and clear or remove stored entries
- replaces old chunks when a document with the same `doc_id` is uploaded again
- filters out weaker retrieval matches by distance
- deduplicates repeated chunks and limits over-representation from a single source
- returns formatted context with source and chunk metadata

## Running The Stack

The current Docker setup lives in [`chat-ui/docker-compose.yml`](./chat-ui/docker-compose.yml).

### Services

- `vllm`: runs `vllm/vllm-openai:nightly`
- `chat-ui`: builds from [`chat-ui/dockerfile`](./chat-ui/dockerfile)

### Default Model

- `Qwen/Qwen3-14B`

### Ports

- `8000`: vLLM API
- `8501`: Streamlit UI

### Mounted Local Directories

- `G:/local-ai/vllm/cache:/root/.cache/huggingface`
- `G:/local-ai/docs:/docs`
- `G:/local-ai/chroma_db:/chroma_db`
- `G:/local-ai/chat_history:/chat_history`

These mounts mean the current compose file is tailored to this Windows machine layout and may need small path changes on another system.

## Configuration

The app currently uses these environment variables:

- `VLLM_API_BASE` default: `http://vllm:8000/v1`
- `MODEL_NAME` default: `Qwen/Qwen3-14B`
- `DOCS_DIR` default: `/docs`
- `CHROMA_DB_PATH` default: `/chroma_db`
- `CHAT_HISTORY_DIR` default: `/chat_history`

The vLLM service is also configured with:

- `--dtype bfloat16`
- `--gpu-memory-utilization 0.95`
- `--max-model-len 16384`
- `--max-num-batched-tokens 16384`
- `--max-num-seqs 16`

For `Qwen3`, the app currently uses non-thinking chat defaults aimed at responsive general use:

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `presence_penalty=1.5`
- `chat_template_kwargs.enable_thinking=false`

## Python Dependencies

[`chat-ui/requirements.txt`](./chat-ui/requirements.txt) currently includes:

- `streamlit`
- `requests`
- `openai`
- `chromadb`
- `sentence-transformers`
- `pypdf`
- `python-dateutil`
- `pytest`

The current app code uses `requests` for vLLM calls; `openai` is installed but not currently used in [`chat-ui/app.py`](./chat-ui/app.py).

## Development Notes

- `docs/`, `chroma_db/`, `chat_history/`, and `vllm/cache/` are runtime data directories, not source directories.
- The repo is currently centered on the `chat-ui` application; there is no separate backend service beyond the compose-managed stack.
- The current Docker setup is still Windows-path-oriented because the compose file mounts `G:/local-ai/...` host directories.

## License

MIT License. See [LICENSE](./LICENSE).
