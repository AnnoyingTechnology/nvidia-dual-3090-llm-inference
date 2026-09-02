# NVIDIA dual RTX 3090 local inference

This repository records the quality-gated optimization of Qwen3.8-27B for two
24 GiB RTX 3090 cards on PCIe 3.0 x16. A one-card, non-speculative control
produced 34.4 tok/s. The selected vLLM service sustains 163.2 tok/s on the
full-context/cache-enabled fixture and reaches 180.3 tok/s on the fixed 225 W
power-sweep cell: **4.7x the one-card control in the production profile and
5.2x on the power cell**.

The result uses a W4A16 target, a W4A16 DFlash2 k=7 drafter, TP=2, BF16 KV,
automatic prefix caching, CPU-offloaded vision, four request slots and the
native 262,144-token server limit. It deliberately avoids quality-affecting
INT8 activation and lower-precision KV paths.

## Result

The live service exposes `qwen3.8-27b` on port `19622` on every host IPv4
interface. Local clients use `http://127.0.0.1:19622/v1`; LAN clients use
`http://<host-address>:19622/v1`. The endpoint is intentionally unauthenticated
and must remain restricted to trusted networks.

| Selected setting | Value |
|---|---|
| Physical GPUs | Two RTX 3090, PCIe 3.0 x16 |
| Tensor parallelism | 2 |
| Target | `Qwen3.8-27B-W4A16-AutoRound-fast` |
| Speculator | DFlash2 W4A16, 7 draft tokens |
| Context | 262,144 server; 253,952 input + 8,192 output for OpenCode |
| KV cache | BF16, 360,791 reported tokens |
| Prefix cache | Enabled, Mamba `align` mode |
| Vision | One image/prompt, CPU-offloaded tower, 2,048 image-token cap |
| Scheduler | 2,048 batched tokens, 4 sequences, async |
| Power | 225 W/card on the inference pair |

The main performance and cache cells were measured at 200 W/card before the
latency-biased cap was applied. Power-cell results are explicitly marked.

| Measurement | Result |
|---|---:|
| Single stream, production profile, approximately p512/g512 | **163.2 tok/s** |
| Two distinct 4K streams | **241.8 aggregate tok/s** |
| Four distinct 4K streams | **309.8 aggregate tok/s** |
| Decode power cell at 225 W, p565/g512 | **180.3 tok/s** |
| Cold prefill power cell at 225 W, p8,221/g8 | **1,570.6 tok/s** |
| Cold 100,040-token retrieval | **86.67 s** |
| Repeated 100K retrieval | **1.07 s**, 99,456 cached tokens |
| OpenCode-style tool follow-up | **0.64 s**, 15,680 cached tokens |
| Real `opencode-cache` continuation | **17,920 cache-read tokens**, exact `CACHE_OK` |
| Synthetic vision canary | Exact `RED, BLUE` |

The concurrency cells completed without preemption. Four users are viable;
one user remains the latency priority, and two users are the best interactive
compromise.

Decode and prefill were swept independently because their power knees can
differ. They converged on the same useful region here. Values are medians and
all output hashes remained identical across caps.

| Cap/card | Decode p565/g512 | Prefill p8,221/g8 |
|---:|---:|---:|
| 125 W | 53.4 tok/s | — |
| 150 W | 102.9 tok/s | 966.9 tok/s |
| 175 W | 154.9 tok/s | 1,349.2 tok/s |
| 200 W | 172.7 tok/s | 1,499.0 tok/s |
| **225 W** | **180.3 tok/s** | **1,570.6 tok/s** |
| 250 W | 184.3 tok/s | 1,618.1 tok/s |
| 275 W | 186.9 tok/s | — |
| 300 W | 188.1 tok/s | 1,678.4 tok/s |
| 350 W | 189.0 tok/s | — |
| 400 W | 189.0 tok/s | — |

The best board-energy efficiency is at 175 W/card. Moving to 200 W adds roughly
11% throughput with almost no efficiency loss. The applied 225 W latency knee
adds another 4–5% for about 11% more board power. Above it, returns collapse.

## Bandwidth reference and remaining headroom

The power plateau is the cleanest measured upper bound for this machine. On the
fixed decode cell, 225 W sustains 180.3 tok/s against 189.0 tok/s at 350–400 W:
**95.4% of the observed ceiling**. Recovering the remaining 4.8% would raise
pair board power from 434.5 W to roughly 601–603 W. On prefill, 225 W retains
93.6% of the 300 W result while using 22.7% less board power.

TP=2 is worthwhile despite the lack of GPU P2P/NVLink: non-speculative decode
rose from 34.4 tok/s on one card to 69.1 tok/s on two. The stripped
65K/cache-off profile reached 169.9 tok/s, but the selected production profile
accepts about a 4% short-prompt cost to retain native context and mandatory
prefix reuse. No credible large lossless gain remains above the 225 W point.

## Optimization ladder

| Accepted decision | Measured contribution | Integrity boundary |
|---|---:|---|
| One x16 card to TP=2, speculation off | 34.4 → 69.1 tok/s, **2.01x** | Same W4A16 target and fixture |
| MTP4 target speculation | 69.1 → 154.8 tok/s, **2.24x** | Target verifies emitted tokens |
| DFlash2 k=7 on the stripped tuning profile | 154.8 → 169.9 tok/s, **+9.8%** | Different production envelope; not multiplied into the final result |
| Native 262K context plus automatic prefix caching | 169.9 → 163.2 tok/s, **-3.9%** | Required serving contract |
| SHA-style cache isolation and append-only reuse | 100K TTFT 86.67 → 1.07 s, **-98.8%** | Reuses only identical prefixes |
| Two and four concurrent streams | 241.8 and 309.8 aggregate tok/s | Zero measured preemptions |
| Separate prefill/decode power sweeps | Selected 225 W latency knee | Same prompt, output and hash at every cap |
| CPU-offloaded vision tower | Full 360,791-token KV pool retained | Exact synthetic image canary |

The percentages are not multiplied: several rows use different controlled
fixtures. Speculative modes also showed sparse greedy hash drift relative to
speculation-off, so they are not described as empirically bit-identical even
though each tested mode was repeat-stable.

## Correctness breakthrough

The optimized checkpoint already contained the complete vision tower and
projector, but the initial validation launch inherited `--language-model-only`.
Enabling the launcher's `VISION=1` path removed that flag, kept the approximately
0.85 GiB tower CPU-offloaded, preserved the 360,791-token KV pool and produced
the exact `RED, BLUE` result on a generated two-color image. The configuration
gap is now covered by a reproducible vision canary.

The vision-enabled production profile passed the full 12/12 API battery:
greedy and seeded determinism, log probabilities, multiple completions, stop
strings, structured JSON, penalties, streaming, thinking, prompt logprobs and a
20K prompt. A real multimodal request correctly identified a generated red/blue
image. The server reports the same 360,791-token KV capacity with vision enabled.

Prefix caching is part of the contract, not an optional benchmark flag. Stable
system/project content, tool schemas and early messages must remain byte-stable;
new conversation and tool turns are appended. The 100K needle test returned the
same exact answer cold and warm while reusing 99,456 tokens. The synthetic tool
flow reused 15,680 tokens and preserved the exact tool call and result.

OpenCode cache-preserving compaction is a separate client behavior. The
`opencode-cache` launcher enables `compaction.preserve_prefix_cache`; the server
cannot recover cache reuse if a client rewrites the early prompt. A live
`opencode-cache` session against this service rendered a 20,863-token cold
agent prompt; its append-only continuation reported 17,920 cache-read tokens
and returned the exact requested canary.

## What might still be left

1. Install a minimal vLLM systemd unit after the interactive profile is
   accepted; the current process is manually launched and will not survive a
   reboot.
2. Run a real long OpenCode session through cache-preserving compaction and
   confirm cached-token accounting after compaction.
3. Convert the host to headless operation after client acceptance. GNOME/GDM
   mainly consume RAM and the excluded x8 card's VRAM, so this is cleanup rather
   than a large inference-speed gain.
4. Evaluate Qwen3.8-Flash-Next later as a separate model/quality campaign.

Further full-TDP sweeps, dynamic prefill/decode caps, INT8 activations and
lower-bit KV are not justified for this production profile.

## Quick operations

Health, model and vision checks:

```bash
curl -fsS http://127.0.0.1:19622/health
curl -fsS http://127.0.0.1:19622/v1/models
python3 scripts/vision_canary.py
```

## Production vLLM invocation

The launcher resolves the selected profile to this material vLLM command and
environment:

```bash
CUDA_VISIBLE_DEVICES=1,2 \
VLLM_VISION_CPU_OFFLOAD_GB=1 \
VLLM_DFLASH2_LOOKUP=1 \
VLLM_SPEC_DECODE_ATTN=1 \
VLLM_SPEC_DECODE_ATTN_QMAX=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve /home/ai/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound-fast \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 \
  --port 19622 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --api-server-count 1 \
  --limit-mm-per-prompt '{"image":{"count":1}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":2097152}}' \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config \
    '{"method":"dflash","model":"/home/ai/qwen-serving/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7}' \
  --compilation-config \
    '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --tensor-parallel-size 2
```

Model and memory settings:

| Setting | Purpose |
|---|---|
| W4A16 target path | Uses the optimized target with quantized embedding, LM head and MTP components. |
| `--max-model-len 262144` | Exposes the native server context; OpenCode reserves 8,192 tokens for output. |
| `--gpu-memory-utilization 0.93` | Produces an 11.68 GiB KV allocation per worker and 360,791 reported tokens. |
| `--kv-cache-dtype bfloat16` | Avoids the quality risk of FP8 or lower-bit KV. |
| `--mamba-ssm-cache-dtype float16` | Provides the capacity required by the production context, with the documented quality boundary. |
| `VLLM_VISION_CPU_OFFLOAD_GB=1` | Keeps the vision tower out of persistent GPU memory while preserving image support. |

Scheduling and agent-session settings:

| Setting | Purpose |
|---|---|
| `--tensor-parallel-size 2` | Uses the two PCIe 3.0 x16 cards; TP traffic crosses host PCIe because P2P is unavailable. |
| `--max-num-seqs 4` | Allows up to four active streams while retaining the single-user priority. |
| `--max-num-batched-tokens 2048` | Selected over 4,096, which regressed the 100K cold-prefill fixture by 4.4%. |
| `--async-scheduling` | Keeps the validated low-latency scheduler path. |
| DFlash2 k=7 | Provides the best selected single-user path without k=15's TP=2 regression. |
| Split-KV verify environment | Sizes and enables the patched speculative verify kernel for an eight-token block. |

API, cache and vision settings:

| Setting | Purpose |
|---|---|
| `--host 0.0.0.0 --port 19622` | Matches B70's host port and exposes the service on LAN. |
| `--served-model-name qwen3.8-27b` | Provides the stable OpenCode model ID. |
| `--enable-prefix-caching --mamba-cache-mode align` | Reuses aligned full-attention/Mamba prefixes. |
| `--enable-prompt-tokens-details` | Returns cached-token accounting used by the canaries and OpenCode validation. |
| `--limit-mm-per-prompt ...` | Enables one image per request and caps startup encoder profiling at 2,048 image tokens. |
| Reasoning and tool parsers | Expose Qwen reasoning plus automatic native tool calls. |
| No API-key option or environment | Matches B70's intentionally unauthenticated trusted-LAN service. |

Power-cap state:

```bash
systemctl status nvidia-power-limit.service
nvidia-smi -i 1,2 --query-gpu=index,enforced.power.limit --format=csv,noheader
```

The endpoint is unauthenticated like the B70 service. Restrict LAN access with
the trusted-network boundary. The current vLLM process has no automatic restart
policy; see [operations](docs/operations.md) before rebooting or stopping it.

## Model and quality contract

The selected target source revision is
`1f05c441c4e64ae0549de44fa9ea5a6d43610314`; the DFlash2 W4A16 revision is
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. The target uses W4A16 weights,
quantized embedding/lm_head/MTP components, BF16 KV and FP16 Mamba state. Those
are quality-relevant boundaries relative to an unquantized BF16 reference.

| Target | English PPL | Danish PPL | Code PPL | Aggregate PPL | GSM8K |
|---|---:|---:|---:|---:|---:|
| Stock optimized W4A16 | 10.765 | 10.908 | 3.132 | **8.1355** | **96.0%** |
| Huihui abliterated W4A16 | 10.769 | 10.934 | 3.156 | 8.1589 | 95.5% |

Huihui was slightly faster on sampled prompts, but aggregate perplexity
regressed 0.29% and GSM8K lost one item out of 200. It remains an optional
uncensored profile rather than the default under the no-regression rule.

The server exposes the native 262,144-token limit and reports capacity for one
maximum-length request plus margin. OpenCode reserves 8,192 output tokens and
therefore declares 253,952 input tokens. Exact cache retrieval is proven at
100K; the absolute input boundary has not received the B70 repository's same
253K completion campaign and is not claimed as independently proven here.

## Documentation

Start with:

- [Benchmarks and quality](docs/benchmarks-and-quality.md): performance,
  concurrency, cache and quality evidence.
- [Architecture](docs/architecture.md): hardware, revisions, runtime, vision and
  quality boundaries.
- [Power efficiency](docs/power-efficiency.md): separate decode and prefill
  sweeps, selection and rollback.
- [Operations](docs/operations.md): startup, validation, cache behavior, power
  persistence and planned headless conversion.
- [References](docs/references.md): upstream sources and pinned model links.

Raw JSON evidence is under `results/`. Model weights, virtual environments,
logs and credentials are intentionally excluded.
