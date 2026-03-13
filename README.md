# Local AI Server (vLLM + Streamlit)

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-brightgreen)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A fully local AI assistant stack powered by GPU-accelerated inference and a lightweight web UI.  
Provides a **ChatGPT-style interface** with optional document search (RAG). Designed for high-end local machines but works on smaller setups with smaller models.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/local-ai-server.git
cd local-ai-server/chat-ui
```

### 2. Prepare vLLM cache folder (Windows, G: drive)

```powershell
mkdir G:\AI\vllm\cache
```

### 3. Start the LLM server

Optimized for RTX 5090, 64GB RAM, running Qwen2.5-32B-Instruct in bf16 to prevent memory issues:

```powershell
docker run --gpus all `
  -p 8000:8000 `
  -v G:\local-ai\vllm\cache:/root/.cache/huggingface `
  vllm/vllm-openai:nightly `
  --model Qwen/Qwen2.5-7B-Instruct `
  --dtype bfloat16 `
  --gpu-memory-utilization 0.70 `
  --max-model-len 8192
```

API endpoint: `http://localhost:8000/v1`  
Test it:

```powershell
curl http://localhost:8000/v1/models
```

### 4. Start the Chat UI

```powershell
docker compose up --build
```

Open in your browser: `http://localhost:8501`

---

## Architecture

```
         ┌─────────────┐
         │   Browser   │
         └─────┬───────┘
               │
               ▼
      ┌─────────────────┐
      │ Streamlit Chat  │
      │ UI (Docker)     │
      └─────┬───────────┘
            │
            ▼
      ┌─────────────┐
      │  vLLM LLM   │
      │  Server     │
      └─────┬───────┘
            │
            ▼
       Local GPU
```

Optional document search:

```
Streamlit UI → Vector DB (Chroma) → Context → vLLM → GPU
```

---

## Features

- ChatGPT-style interface  
- Streaming responses  
- Conversation memory  
- Document search (RAG)  
- Fully local inference  
- GPU acceleration  
- Dockerized deployment  

---

## Tech Stack

| Component              | Purpose                                    |
|------------------------|--------------------------------------------|
| vLLM                   | High-performance LLM inference server      |
| Streamlit              | Chat web interface                          |
| Docker                 | Containerized deployment                    |
| Chroma                 | Vector database for document search        |
| Sentence Transformers  | Embedding generation                        |

---

## Hardware Example

Tested on:

- GPU: NVIDIA GeForce RTX 5090  
- CPU: AMD Ryzen 7 9800X3D  
- RAM: 64GB  
- OS: Windows 11 Pro  
- Storage: G: drive for AI models and cache

Lower-end systems can use smaller models for compatibility.

---

## Project Structure

```
local-ai-server
│
├── chat-ui
│   ├── app.py
│   ├── rag.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs
│   └── setup.md
│
├── .gitignore
└── README.md
```

Local runtime directories (not included in repo):

```
G:\AI
│
├── vllm
│   └── cache
│
└── local-ai-server
```

---

## Document Search (RAG)

```python
from rag import add_document

text = open("manual.txt").read()
add_document(text)
```

---

## Recommended Models

- Qwen2.5-32B-Instruct  
- Llama-3.1-70B-Instruct (quantized)  
- Mixtral 8x22B  

---

## Development Commands

Rebuild UI container:

```bash
docker compose up --build
```

Stop containers:

```bash
docker compose down
```

Check running containers:

```bash
docker ps
```

---

## Screenshots

*Add your own screenshots here.*

![UI Screenshot](docs/screenshot-placeholder.png)

---

## Future Improvements

- Voice assistant integration (Whisper + Piper)  
- Web search agents  
- Multi-model routing  
- Local image generation (ComfyUI / Stable Diffusion)  
- Long-term vector memory  
- Authentication / multi-user access  

---

## License

MIT License

---

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm)  
- [Streamlit](https://streamlit.io/)  
- [Chroma](https://www.trychroma.com/)  
- [Hugging Face](https://huggingface.co/)

