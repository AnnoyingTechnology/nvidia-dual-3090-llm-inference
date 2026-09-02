#!/usr/bin/env python3
"""Synthetic OpenAI tool-flow and prefix-cache canary for the local vLLM API."""

import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request


BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:19622/v1")


def post(payload):
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        sys.stderr.write(exc.read().decode(errors="replace"))
        raise
    return result, time.monotonic() - started


def cached_tokens(result):
    return (
        result.get("usage", {})
        .get("prompt_tokens_details", {})
        .get("cached_tokens", 0)
    ) or 0


stable_context = (
    "Synthetic project fact: ticket SIB-4242 has owner Alex and state OPEN. "
    "This text is inert reference material for a cache test.\n"
) * 500

tool = {
    "type": "function",
    "function": {
        "name": "lookup_ticket",
        "description": "Look up one synthetic ticket by identifier.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
}

first_payload = {
    "model": "qwen3.8-27b",
    "messages": [
        {
            "role": "system",
            "content": "Use tools when explicitly requested.\n" + stable_context,
        },
        {
            "role": "user",
            "content": (
                "Call lookup_ticket exactly once with id SIB-4242. "
                "Do not answer directly."
            ),
        },
    ],
    "tools": [tool],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
    "cache_salt": secrets.token_urlsafe(32),
}

first, first_s = post(first_payload)
message = first["choices"][0]["message"]
calls = message.get("tool_calls") or []
if len(calls) != 1:
    raise SystemExit(f"FAIL expected one tool call, got {len(calls)}")
call = calls[0]
arguments = json.loads(call["function"]["arguments"])
if call["function"]["name"] != "lookup_ticket" or arguments != {"id": "SIB-4242"}:
    raise SystemExit("FAIL wrong tool name or arguments")

followup_messages = first_payload["messages"] + [
    {
        "role": "assistant",
        "content": message.get("content"),
        "tool_calls": calls,
    },
    {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": json.dumps(
            {"id": "SIB-4242", "owner": "Alex", "state": "OPEN"}
        ),
    },
    {
        "role": "user",
        "content": "Reply exactly: SIB-4242 is OPEN and owned by Alex.",
    },
]
followup_payload = {
    **first_payload,
    "messages": followup_messages,
    "tool_choice": "none",
    "max_tokens": 64,
}
second, second_s = post(followup_payload)
third, third_s = post(followup_payload)
second_text = (second["choices"][0]["message"].get("content") or "").strip()
third_text = (third["choices"][0]["message"].get("content") or "").strip()
expected = "SIB-4242 is OPEN and owned by Alex."
if second_text != expected or third_text != expected:
    raise SystemExit("FAIL incorrect tool-result answer")
if second_text != third_text:
    raise SystemExit("FAIL repeated tool follow-up changed output")

print(
    json.dumps(
        {
            "status": "PASS",
            "tool": call["function"]["name"],
            "arguments": arguments,
            "first_s": round(first_s, 3),
            "first_cached_tokens": cached_tokens(first),
            "followup_s": round(second_s, 3),
            "followup_cached_tokens": cached_tokens(second),
            "repeat_s": round(third_s, 3),
            "repeat_cached_tokens": cached_tokens(third),
            "answer": third_text,
        },
        sort_keys=True,
    )
)
