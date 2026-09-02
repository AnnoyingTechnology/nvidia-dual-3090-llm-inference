#!/usr/bin/env python3
"""Hash a compact deterministic response suite without printing response text."""
import hashlib
import json
import os
import urllib.request


PORT = os.environ.get("PORT", "19622")
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b-abliterated")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PROMPTS = [
    "Explain why a mutex does not by itself prevent deadlock. Give three concise points.",
    "Write a Python function that returns the length of the longest increasing subsequence in O(n log n).",
    "Solve exactly: If 3x + 7 = 52, what is x? Show the calculation.",
    "Svar på dansk med præcis to sætninger: Hvorfor er regelmæssige sikkerhedskopier vigtige?",
    "Return only valid JSON with keys name, ports, and enabled for a service named inference using ports 8000 and 8001.",
]


def ask(prompt: str) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"]


outputs = [ask(prompt) for prompt in PROMPTS]
print(json.dumps({
    "suite": hashlib.sha256("\0".join(outputs).encode()).hexdigest(),
    "items": [
        {"length": len(output), "sha256": hashlib.sha256(output.encode()).hexdigest()}
        for output in outputs
    ],
}))
