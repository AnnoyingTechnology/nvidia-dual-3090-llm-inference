#!/usr/bin/env python3
"""Measure cold, exact-repeat and append-only prefix-cache behavior."""
import json
import os
import time
import urllib.request


PORT = os.environ.get("PORT", "18020")
KEY_PATH = os.environ.get("KEY_PATH", "/home/ai/qwen-serving/api_key.txt")
KEY = open(KEY_PATH, encoding="utf-8").read().strip()
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def ask(messages, max_tokens=32):
    body = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + KEY,
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1200) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": details.get("cached_tokens", 0),
        "content": result["choices"][0]["message"]["content"],
    }


document = "".join(
    f"Record {index:05d}: alpha beta gamma delta epsilon zeta eta theta.\n"
    for index in range(2000)
)
system = "You are a precise coding assistant. Follow instructions literally."
first_messages = [
    {"role": "system", "content": system},
    {
        "role": "user",
        "content": document + "\nReply with exactly: CACHE-READY",
    },
]

cold = ask(first_messages)
warm = ask(first_messages)
continued = ask(first_messages + [
    {"role": "assistant", "content": warm["content"]},
    {"role": "user", "content": "Reply with exactly: CACHE-CONTINUED"},
])

print(json.dumps({
    "cold": {key: value for key, value in cold.items() if key != "content"},
    "warm": {key: value for key, value in warm.items() if key != "content"},
    "continued": {key: value for key, value in continued.items() if key != "content"},
    "same_output": cold["content"] == warm["content"],
    "cold_output": cold["content"],
    "continued_output": continued["content"],
}))
