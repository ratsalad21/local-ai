import json
from typing import Any

import requests

from config import (
    APPROX_CHARS_PER_TOKEN,
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_MESSAGES,
    MIN_OUTPUT_TOKENS,
    REQUEST_TIMEOUT,
    STATUS_TIMEOUT,
    VLLM_API_BASE,
    VLLM_MAX_MODEL_LEN,
)


def get_model_server_status() -> dict[str, Any]:
    status = {
        "reachable": False,
        "models": [],
        "error": None,
    }

    try:
        response = requests.get(f"{VLLM_API_BASE}/models", timeout=STATUS_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        status["reachable"] = True
        status["models"] = [item.get("id", "unknown") for item in payload.get("data", [])]
        return status
    except Exception as exc:
        status["error"] = str(exc)
        return status


def build_api_messages(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages[-MAX_HISTORY_MESSAGES:]:
        api_messages.append(
            {
                "role": msg["role"],
                "content": msg["content"],
            }
        )

    return api_messages


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + APPROX_CHARS_PER_TOKEN - 1) // APPROX_CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    # Keep the heuristic simple and biased slightly high to avoid vLLM token-limit errors.
    return sum(estimate_text_tokens(msg.get("content", "")) + 12 for msg in messages)


def fit_messages_to_budget(
    system_prompt: str,
    messages: list[dict[str, Any]],
    requested_output_tokens: int,
) -> tuple[list[dict[str, str]], int]:
    output_tokens = max(MIN_OUTPUT_TOKENS, min(requested_output_tokens, VLLM_MAX_MODEL_LEN // 2))
    budget = max(MIN_OUTPUT_TOKENS, VLLM_MAX_MODEL_LEN - output_tokens - 32)
    recent_messages = messages[-MAX_HISTORY_MESSAGES:]

    while recent_messages:
        api_messages = build_api_messages(system_prompt, recent_messages)
        if estimate_messages_tokens(api_messages) <= budget:
            return api_messages, output_tokens
        recent_messages = recent_messages[2:]

    return build_api_messages(system_prompt, messages[-1:]), output_tokens


def build_retrieval_system_prompt(base_prompt: str, context: str) -> str:
    if not context:
        return base_prompt

    return (
        f"{base_prompt}\n\n"
        "You may use the retrieved context below if it is relevant. "
        "When the answer depends on that context, mention the source file names naturally in your answer.\n\n"
        f"Retrieved context:\n{context[:MAX_CONTEXT_CHARS]}"
    )


def stream_chat_completion(payload: dict[str, Any], stream_state: dict[str, Any] | None = None):
    full_response = ""

    with requests.post(
        f"{VLLM_API_BASE}/chat/completions",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        response.raise_for_status()

        # vLLM streams server-sent events where each "data:" line contains the next delta.
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            data = line[len(b"data: ") :]
            if data == b"[DONE]":
                break

            try:
                obj = json.loads(data.decode("utf-8"))
                choice = obj["choices"][0]
                delta = choice["delta"].get("content", "")
                finish_reason = choice.get("finish_reason")

                if finish_reason and stream_state is not None:
                    stream_state["finish_reason"] = finish_reason

                if delta:
                    full_response += delta
                    yield full_response
            except Exception:
                continue
