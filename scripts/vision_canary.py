#!/usr/bin/env python3
"""Exercise the live vision projector with a deterministic synthetic image."""

import base64
import json
import os
import struct
import urllib.request
import zlib


BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:18020/v1")
API_KEY_FILE = os.environ.get(
    "QWEN_API_KEY_FILE", "/home/ai/qwen-serving/api_key.txt"
)


def png_chunk(kind, payload):
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


width, height = 64, 32
row = b"\x00" + (b"\xff\x00\x00" * 32) + (b"\x00\x00\xff" * 32)
png = b"\x89PNG\r\n\x1a\n"
png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
png += png_chunk(b"IDAT", zlib.compress(row * height))
png += png_chunk(b"IEND", b"")
image_url = "data:image/png;base64," + base64.b64encode(png).decode()

with open(API_KEY_FILE, encoding="utf-8") as handle:
    api_key = handle.read().strip()

payload = {
    "model": "qwen3.8-27b",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {
                    "type": "text",
                    "text": (
                        "Name the color on the left and the color on the right. "
                        "Reply exactly: LEFT, RIGHT"
                    ),
                },
            ],
        }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "chat_template_kwargs": {"enable_thinking": False},
}

request = urllib.request.Request(
    BASE_URL + "/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)

answer = (result["choices"][0]["message"].get("content") or "").strip()
passed = answer == "RED, BLUE"
print(json.dumps({"status": "PASS" if passed else "FAIL", "answer": answer}))
raise SystemExit(0 if passed else 1)
