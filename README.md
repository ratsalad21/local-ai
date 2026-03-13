# Local AI Chat Assistant

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-brightgreen)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A fully local AI chat assistant powered by GPU-accelerated inference using vLLM and a lightweight Streamlit web UI.  
Provides a **ChatGPT-style interface** with optional document search (RAG) using Chroma vector database. Designed for high-end local machines but works on smaller setups with smaller models.

---

## Quick Start

### Prerequisites
- Docker with GPU support (NVIDIA Docker)
- At least 16GB RAM (32GB+ recommended)
- NVIDIA GPU with sufficient VRAM (e.g., RTX 3090+ for larger models)

### 1. Clone the repository

```bash
git clone https://github.com/ratsalad21/local-ai.git
cd local-ai/chat-ui
```

### 2. Start the services

The project uses Docker Compose to run both the vLLM inference server and the Streamlit chat UI.

```bash
docker compose up --build
```

This will:
- Start the vLLM server on port 8000 with Qwen2.5-7B-Instruct model
- Start the Streamlit chat UI on port 8501

### 3. Access the chat interface

Open your browser and navigate to: `http://localhost:8501`

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

With RAG enabled:

```
Streamlit UI → Chroma Vector DB → Context → vLLM → GPU
```

---

## Features

- 🐶 **Bonzo AI Assistant**: Friendly local AI chat interface
- 💬 **Streaming responses**: Real-time text generation with typing indicator
- 🧠 **Conversation memory**: Maintains chat history across the session
- 📄 **Document search (RAG)**: Upload and search through documents (txt/md/pdf) for context
- 🎨 **Light/Dark themes**: Toggle UI themes via the sidebar
- 📋 **Copy and export**: Copy individual messages or export full chat as Markdown
- 👍👎 **Reactions & regeneration**: React to responses and regenerate the last answer
- 🔒 **Fully local**: No data sent to external servers
- 🚀 **GPU acceleration**: Optimized for NVIDIA GPUs
- 🐳 **Dockerized**: Easy deployment and isolation

---

## Tech Stack

| Component              | Purpose                                    |
|------------------------|--------------------------------------------|
| vLLM                   | High-performance LLM inference server      |
| Streamlit              | Web-based chat interface                   |
| Docker                 | Containerized deployment                   |
| Chroma                 | Vector database for document embeddings    |
| Sentence Transformers  | Text embedding generation                  |
| PyPDF                  | PDF document processing                    |

---

## Hardware Requirements

**Recommended Setup:**
- GPU: NVIDIA GeForce RTX 3090 or higher
- CPU: Modern multi-core processor
- RAM: 32GB+ DDR4/DDR5
- Storage: SSD with 100GB+ free space for models
- OS: Windows 10/11, Linux, or macOS (with GPU passthrough)

**Minimum Setup:**
- GPU: NVIDIA GPU with 8GB+ VRAM
- RAM: 16GB
- Can run smaller models like Qwen2.5-7B-Instruct

---

## Project Structure

```
local-ai/
│
├── chat-ui/
│   ├── app.py              # Main Streamlit application
│   ├── rag.py              # RAG functionality with Chroma
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile          # Docker image for chat UI
│   ├── docker-compose.yml  # Multi-service Docker setup
│   └── tests/              # Unit tests (pytest)
│
├── vllm/
│   └── cache/              # Hugging Face model cache
│       └── hub/
│           └── models--Qwen--Qwen2.5-*/  # Cached models
│
├── chroma_db/              # Chroma vector database files
│
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Configuration

### Environment Variables

The chat UI can be configured via environment variables:

- `VLLM_API_BASE`: vLLM API endpoint (default: `http://host.docker.internal:8000/v1`)
- `MODEL_NAME`: Model identifier (default: `Qwen/Qwen2.5-7B-Instruct`)

### Changing Models

To use a different model, modify the `command` in `docker-compose.yml`:

```yaml
command: >
  --model Qwen/Qwen2.5-32B-Instruct
  --dtype bfloat16
  --gpu-memory-utilization 0.70
  --max-model-len 8192
```

Available cached models in this repository:
- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct

---

## Document Search (RAG)

Upload text, markdown, or PDF files through the sidebar to enhance the AI's knowledge base.

```python
# Example usage in code
from rag import add_document, query_documents

# Add a document
add_document("Your document text here", doc_id="example.txt")

# Query for relevant context
context = query_documents("What is machine learning?")
```

---

## Development

### Running locally (without Docker)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start vLLM server separately:
```bash
docker run --gpus all -p 8000:8000 -v G:\local-ai\vllm\cache:/root/.cache/huggingface vllm/vllm-openai:nightly --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

### Run tests
```bash
pytest
```

### Rebuilding containers

```bash
docker compose down
docker compose up --build
```

### Checking logs

```bash
# vLLM server logs
docker logs vllm-server

# Chat UI logs
docker logs local-chat-ui
```

---

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure NVIDIA Docker is installed and GPU drivers are up to date.

2. **Port conflicts**: Change ports in `docker-compose.yml` if 8000 or 8501 are in use.

3. **Memory issues**: Reduce `--gpu-memory-utilization` or use a smaller model.

4. **Model download fails**: Check internet connection and Hugging Face access.

### Performance Tips

- Use SSD storage for model cache
- Close other GPU-intensive applications
- Monitor GPU memory usage with `nvidia-smi`
- For larger models, increase `--max-model-len` if needed

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Matt Everhart

---

## Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
- [Streamlit](https://streamlit.io/) for the web interface
- [Chroma](https://www.trychroma.com/) for vector database
- [Qwen](https://github.com/QwenLM/Qwen2.5) for the language models

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

