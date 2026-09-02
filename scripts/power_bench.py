#!/usr/bin/env python3
"""Measure gross GPU-board prefill and decode efficiency across RTX 3090 power caps."""

import argparse
import bisect
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid


GPU_IDS = (1, 2)
NVIDIA_SMI = "/usr/bin/nvidia-smi"
FILLER = (
    "The RTX 3090 has 24 GB of GDDR6X and 82 streaming multiprocessors. "
    "Memory bandwidth is 936 GB/s, which is what decode is bound by. "
)


def gpu_query(fields):
    output = subprocess.check_output(
        [
            NVIDIA_SMI,
            "-i",
            ",".join(map(str, GPU_IDS)),
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        [float(value.strip()) for value in line.split(",")]
        for line in output.strip().splitlines()
    ]


def set_cap(gpu_id, watts):
    subprocess.run(
        [NVIDIA_SMI, "-i", str(gpu_id), "-pl", str(int(watts))],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


class PowerSampler:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                rows = gpu_query(("power.draw", "clocks.sm", "clocks.mem", "temperature.gpu"))
                self.samples.append(
                    {
                        "t": time.monotonic(),
                        "power_w": sum(row[0] for row in rows),
                        "sm_mhz": [row[1] for row in rows],
                        "mem_mhz": [row[2] for row in rows],
                        "temp_c": [row[3] for row in rows],
                    }
                )
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(self.interval)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)


def interpolate(samples, instant, field):
    times = [sample["t"] for sample in samples]
    pos = bisect.bisect_left(times, instant)
    if pos <= 0:
        return samples[0][field]
    if pos >= len(samples):
        return samples[-1][field]
    left, right = samples[pos - 1], samples[pos]
    span = right["t"] - left["t"]
    if span <= 0:
        return right[field]
    fraction = (instant - left["t"]) / span
    return left[field] + fraction * (right[field] - left[field])


def integrate_power(samples, start, end):
    if end <= start or len(samples) < 2:
        return 0.0
    points = [(start, interpolate(samples, start, "power_w"))]
    points.extend(
        (sample["t"], sample["power_w"])
        for sample in samples
        if start < sample["t"] < end
    )
    points.append((end, interpolate(samples, end, "power_w")))
    return sum(
        (right_t - left_t) * (left_w + right_w) / 2
        for (left_t, left_w), (right_t, right_w) in zip(points, points[1:])
    )


def extrema(samples, start, end, field, reducer):
    values = [sample[field] for sample in samples if start <= sample["t"] <= end]
    flattened = [value for group in values for value in group]
    return reducer(flattened) if flattened else None


def make_prompt(target_tokens):
    repeats = max(1, round(target_tokens / 44))
    return (
        FILLER * repeats
        + "\n\nWrite a detailed technical explanation of why memory bandwidth, "
        "not compute, limits single-stream decoding on this hardware. Be thorough."
    )


def run_request(api, prompt, output_tokens, sampler):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        # Keep text identical across caps while forcing a cold cache namespace.
        "cache_salt": "power-sweep-" + uuid.uuid4().hex,
    }
    request = urllib.request.Request(
        api + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    last = None
    usage = {}
    content = []
    finish_reason = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            event = line[6:]
            if event == "[DONE]":
                break
            chunk = json.loads(event)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason") is not None:
                finish_reason = choices[0]["finish_reason"]
            text = choices[0].get("delta", {}).get("content")
            if text:
                now = time.monotonic()
                first = now if first is None else first
                last = now
                content.append(text)
    ended = time.monotonic()
    if first is None or last is None or not usage:
        raise RuntimeError("stream did not return timed content and usage")

    completion_tokens = int(usage["completion_tokens"])
    prompt_tokens = int(usage["prompt_tokens"])
    cached_prompt_tokens = int(
        usage.get("prompt_tokens_details", {}).get("cached_tokens") or 0
    )
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    decode_tokens = max(0, completion_tokens - 1)
    prefill_s = first - started
    decode_s = last - first
    total_energy_j = integrate_power(sampler.samples, started, ended)
    prefill_energy_j = integrate_power(sampler.samples, started, first)
    decode_energy_j = integrate_power(sampler.samples, first, last)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": prefill_s,
        "decode_s": decode_s,
        "total_s": ended - started,
        "prefill_tokens_per_s": uncached_prompt_tokens / prefill_s,
        "decode_tokens_per_s": decode_tokens / decode_s,
        "total_gpu_energy_j": total_energy_j,
        "prefill_gpu_energy_j": prefill_energy_j,
        "decode_gpu_energy_j": decode_energy_j,
        "prefill_tokens_per_gpu_j": uncached_prompt_tokens / prefill_energy_j,
        "decode_tokens_per_gpu_j": decode_tokens / decode_energy_j,
        "average_prefill_gpu_power_w": prefill_energy_j / prefill_s,
        "average_decode_gpu_power_w": decode_energy_j / decode_s,
        "peak_total_gpu_power_w": max(
            (sample["power_w"] for sample in sampler.samples if first <= sample["t"] <= last),
            default=None,
        ),
        "max_temperature_c": extrema(sampler.samples, started, ended, "temp_c", max),
        "median_sm_clock_mhz": extrema(
            sampler.samples, first, last, "sm_mhz", statistics.median
        ),
        "median_mem_clock_mhz": extrema(
            sampler.samples, first, last, "mem_mhz", statistics.median
        ),
        "output_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
        "sample_count": sum(started <= sample["t"] <= ended for sample in sampler.samples),
    }


def median_summary(cap, rows):
    numeric = (
        "ttft_s",
        "decode_s",
        "total_s",
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "total_gpu_energy_j",
        "prefill_gpu_energy_j",
        "decode_gpu_energy_j",
        "prefill_tokens_per_gpu_j",
        "decode_tokens_per_gpu_j",
        "average_prefill_gpu_power_w",
        "average_decode_gpu_power_w",
        "peak_total_gpu_power_w",
        "max_temperature_c",
        "median_sm_clock_mhz",
        "median_mem_clock_mhz",
    )
    summary = {"power_cap_w_per_gpu": cap, "repetitions": len(rows)}
    for field in numeric:
        values = [row[field] for row in rows if row[field] is not None]
        if values:
            summary[field + "_median"] = statistics.median(values)
    summary["all_outputs_identical"] = len({row["output_sha256"] for row in rows}) == 1
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="150,175,200,225,250,275,300,350")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--api", default="http://127.0.0.1:19622/v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    caps = [int(value) for value in args.caps.split(",")]
    limits = gpu_query(("power.min_limit", "power.max_limit", "enforced.power.limit"))
    originals = [row[2] for row in limits]
    for cap in caps:
        for row in limits:
            if not row[0] <= cap <= row[1]:
                raise SystemExit(f"cap {cap} outside supported range {row[0]}..{row[1]}")

    prompt = make_prompt(args.prompt_tokens)
    campaign = {
        "schema": 1,
        "scope": "GPU board telemetry only; excludes CPU, RAM and PSU losses",
        "gpu_ids": list(GPU_IDS),
        "original_power_caps_w": originals,
        "requested_caps_w": caps,
        "prompt_target_tokens": args.prompt_tokens,
        "output_target_tokens": args.output_tokens,
        "results": [],
    }

    sampler = PowerSampler()
    sampler.start()
    try:
        for cap in caps:
            for gpu_id in GPU_IDS:
                set_cap(gpu_id, cap)
            time.sleep(2)
            rows = []
            for repetition in range(1, args.reps + 1):
                row = run_request(args.api, prompt, args.output_tokens, sampler)
                row["repetition"] = repetition
                rows.append(row)
                print(json.dumps({"cap_w": cap, **row}), flush=True)
                time.sleep(0.5)
            campaign["results"].append(
                {"summary": median_summary(cap, rows), "runs": rows}
            )
    finally:
        for gpu_id, original in zip(GPU_IDS, originals):
            set_cap(gpu_id, original)
        sampler.stop()

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(campaign, handle, indent=2)
        handle.write("\n")
    if os.geteuid() == 0:
        directory = os.stat(output_dir)
        os.chown(args.output, directory.st_uid, directory.st_gid)
    print(json.dumps({"status": "PASS", "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
