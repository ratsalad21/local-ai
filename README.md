# Local AI

Local AI is a self-hosted chat stack for running a local assistant with a Streamlit UI, a vLLM inference server, and optional document retrieval through Chroma.

The repo is intentionally small: one app in [`chat-ui`](./chat-ui), a Docker Compose setup for the UI and model server, and local state directories for documents, embeddings, and model cache.

## Overview

This project currently gives you:

- a browser-based local chat interface
- streaming responses from a vLLM OpenAI-compatible endpoint
- document upload for `txt`, `md`, and `pdf`
- persistent local RAG storage with Chroma
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

- keeps chat history in Streamlit session state
- sends chat requests directly to the vLLM HTTP API
- supports a custom system prompt, temperature, and max token controls
- can optionally fetch retrieved document context before generation

The RAG layer in [`chat-ui/rag.py`](./chat-ui/rag.py):

- splits documents into overlapping word chunks
- uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- stores vectors in a Chroma collection named `documents`
- replaces old chunks when a document with the same `doc_id` is uploaded again
- returns formatted context with source and chunk metadata

## Running The Stack

The current Docker setup lives in [`chat-ui/docker-compose.yml`](./chat-ui/docker-compose.yml).

### Services

- `vllm`: runs `vllm/vllm-openai:nightly`
- `chat-ui`: builds from [`chat-ui/dockerfile`](./chat-ui/dockerfile)

### Default Model

- `Qwen/Qwen2.5-7B-Instruct`

### Ports

- `8000`: vLLM API
- `8501`: Streamlit UI

### Mounted Local Directories

- `G:/local-ai/vllm/cache:/root/.cache/huggingface`
- `G:/local-ai/docs:/docs`
- `G:/local-ai/chroma_db:/chroma_db`

These mounts mean the current compose file is tailored to this Windows machine layout and may need small path changes on another system.

## Configuration

The app currently uses these environment variables:

- `VLLM_API_BASE` default: `http://vllm:8000/v1`
- `MODEL_NAME` default: `Qwen/Qwen2.5-7B-Instruct`

The vLLM service is also configured with:

- `--dtype auto`
- `--gpu-memory-utilization 0.92`
- `--max-model-len 4096`
- `--max-num-batched-tokens 16384`
- `--max-num-seqs 16`

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

- `docs/`, `chroma_db/`, and `vllm/cache/` are runtime data directories, not source directories.
- The repo is currently centered on the `chat-ui` application; there is no separate backend service beyond the compose-managed stack.
- The checked-in tests in [`chat-ui/tests/test_rag.py`](./chat-ui/tests/test_rag.py) appear to target an older `rag.py` interface and may need updating before they pass against the current code.

## License

MIT License. See [LICENSE](./LICENSE).
