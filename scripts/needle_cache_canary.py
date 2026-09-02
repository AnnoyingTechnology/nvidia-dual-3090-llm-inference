#!/usr/bin/env python3
"""Cache-isolated long-context retrieval and exact-repeat canary."""

import json
import os
import secrets
import time
import urllib.request


api = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:18020/v1")
key_file = os.environ.get("QWEN_API_KEY_FILE", "/home/ai/qwen-serving/api_key.txt")
with open(key_file, encoding="utf-8") as handle:
    key = handle.read().strip()

target_tokens = int(os.environ.get("TARGET_TOKENS", "100000"))
depth = float(os.environ.get("NEEDLE_DEPTH", "0.9"))
needle = "ZXCVBNM12345"
unit = "All work and no play makes Jack a dull boy. "
filler = unit * int(target_tokens / 11)
offset = int(len(filler) * depth)
prompt = (
    filler[:offset]
    + f"\n\nThe secret passcode is {needle}. Remember it exactly.\n\n"
    + filler[offset:]
    + "\n\nQuestion: what is the secret passcode? Reply with the passcode only."
)

payload = {
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": False},
    "cache_salt": secrets.token_urlsafe(32),
}


def run():
    request = urllib.request.Request(
        api + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    elapsed = time.monotonic() - started
    usage = result.get("usage", {})
    details = usage.get("prompt_tokens_details", {})
    message = result["choices"][0]["message"]
    return {
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "cached_tokens": details.get("cached_tokens", 0) or 0,
        "answer": (message.get("content") or "").strip(),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
    }


cold = run()
warm = run()
passed = needle in cold["answer"] and cold["answer"] == warm["answer"]
print(json.dumps({"status": "PASS" if passed else "FAIL", "cold": cold, "warm": warm}))
raise SystemExit(0 if passed else 1)
