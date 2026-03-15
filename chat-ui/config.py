import os
from pathlib import Path

from dateutil import tz

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://vllm:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/docs"))
CHAT_HISTORY_DIR = Path(os.getenv("CHAT_HISTORY_DIR", "/chat_history"))

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_CONTEXT_CHARS = 2000
MAX_HISTORY_MESSAGES = 8
REQUEST_TIMEOUT = 300
STATUS_TIMEOUT = 5
RETRIEVAL_K = 5
DOC_PREVIEW_CHARS = 3000
DOC_SEARCH_RESULTS = 5
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_PRESENCE_PENALTY = 1.5
DEFAULT_ENABLE_THINKING = False
MIN_OUTPUT_TOKENS = 64
DEFAULT_OUTPUT_TOKENS = 512
APPROX_CHARS_PER_TOKEN = 4

DOCS_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

EASTERN = tz.gettz("America/New_York")
