# Three-GPU evaluation

## Decision

The three-GPU optimization campaign was closed on 2026-09-10. None of the
tested configurations produced a material net improvement while preserving the
single-endpoint contract and keeping regressions below approximately 15%.

Production remains TP=2 on physical GPUs 1 and 2, with DFlash2 k=7, BF16 KV,
the 262,144-token server context, automatic prefix caching and CPU-offloaded
vision. Physical GPU 0 remains outside vLLM.

The only technically admissible alternative was DFlash2 k=3. It gained 5.8%
aggregate output throughput at four concurrent requests, but lost 6.1%
single-stream output throughput and increased single-stream TPOT by 10.8%.
That is a poor operational trade for a service whose priority is interactive
latency.

## Required serving contract

Candidates had to preserve all of the following:

- one OpenAI-compatible endpoint able to handle every request;
- vision, full native context and BF16 KV;
- automatic prefix caching, without requiring client-side routing or affinity;
- speculative decoding;
- no material quality change;
- no score regression greater than approximately 15%.

Separate endpoints and external routing were rejected because the current
client cannot distribute requests or provide sticky affinity. Prefix caches are
engine-local, so transparent distribution would also make cache reuse
unpredictable.

## Test platform

- Three 24 GiB RTX 3090 cards.
- Physical GPUs 1 and 2: PCIe 3.0 x16, used by production TP=2.
- Physical GPU 0: PCIe x8, also hosting the desktop display.
- No usable GPU P2P/NVLink path; collectives cross host PCIe.
- Inference pair capped at 225 W/card; GPU 0 capped at 200 W.
- Huihui Qwen3.8-27B W4A16 target and W4A16 DFlash2 drafter.

Short decode comparisons used identical random prompts and seeds: 4,096 input
tokens, 512 generated tokens, greedy decoding, at concurrency 1 and 4. Prefill
used 8,192 input tokens and one generated token. K3 and the K7 control decode
cells were each run twice. Power values are NVIDIA board telemetry summed over
all three GPUs during matched runs; they exclude CPU, RAM, storage and PSU
losses.

After the initial TP/PP/DP investigation, an independent GPT-6 Astra review at
medium reasoning effort selected two bounded follow-ups: a DFlash verify-block
sweep (k=3/5/9 against k=7) and a FlashAttention/FlashInfer A/B. Both were then
tested on the live host under the same serving contract.

## DFlash verify-block sweep

| Configuration | C1 output tok/s | C4 output tok/s | C1 TPOT | 8K prefill tok/s | Result |
|---|---:|---:|---:|---:|---|
| K7 production control | 82.57 | 118.64 | 7.02 ms | 1,571 | Selected |
| K3 | 77.56 | 125.50 | 7.78 ms | 1,558 | Rejected: marginal aggregate gain |
| K5 | 81.74 | 124.43 | 7.13 ms | Not run | Rejected: below 5% gain |
| K9 adaptive maximum | 65.00 | Aborted | 10.32 ms | Not run | Rejected: C1 regression |

K3 relative to K7:

- C1 output throughput: -6.1%.
- C1 TPOT: +10.8%.
- C4 aggregate output throughput: +5.8%.
- 8K prefill: -0.8% against the two-run K7 mean.
- Four-request mean TPOT: effectively unchanged.
- Universal API battery: 12/12 passed, including a 20K-token prompt.

K5 kept C1 effectively flat, but its single C4 run improved aggregate output by
only 4.9% against the K7 mean. K9 was stopped after its C1 screen: output fell
approximately 21% and TPOT rose approximately 47%. The installed lookup path
already supports a variable K9 maximum over its trained seven-token draft
block; it did not make K9 competitive on this workload.

### Power cost

| Matched workload | K7 all-GPU average | K3 all-GPU average | Peak, both |
|---|---:|---:|---:|
| C1 | 298.4 W | 298.2 W | 475 W |
| C4 | 360.1 W | 359.3 W | 475 W |

No power increase was measurable. K3's C4 tokens per GPU-board joule improved
by approximately 6%, matching its throughput gain, while its C1 efficiency fell
by approximately 6%. GPU 0 contributed roughly 20 W of desktop/idle draw to
these totals. Removing the desktop may save that idle draw, but does not improve
the TP=2 compute path.

## Attention backend

The production TP=2/K7 profile was started with FlashInfer instead of
FlashAttention while retaining context, BF16 KV, vision, prefix caching and
speculation. It initialized successfully but failed the first performance gate:

| Backend | C1 output tok/s | C1 TTFT | C1 TPOT | Result |
|---|---:|---:|---:|---|
| FlashAttention control | 82.57 | 2.61 s | 7.02 ms | Selected |
| FlashInfer | 54.91 | 2.62 s | 13.12 ms | Rejected |

FlashInfer left TTFT unchanged but reduced output throughput by approximately
33% and increased TPOT by approximately 87%. Further testing was aborted.

## Three-way execution layouts

### Tensor parallelism

TP=3 is invalid for this target because its 32 attention heads are not divisible
by three. Better PCIe bandwidth does not remove this model-topology constraint.

### Pipeline parallelism

PP=3 without speculation was functional and improved prefill and sufficiently
batched throughput:

| PP=3, speculation off | Result |
|---|---:|
| Decode C1 / C2 / C4 | 49.8 / 90.9 / 173.3 tok/s |
| Cold 8K prefill | 2,154 tok/s |
| Cold 32K prefill | 2,553 tok/s |

The comparable TP=2 speculation-off prefill figures were 1,598 tok/s at 8K and
1,488 tok/s at 32K. PP=3 therefore demonstrated useful prefill scaling, but the
DFlash path does not support pipeline parallelism. Losing mandatory speculation
made C1 decode unacceptable, so PP=3 was rejected.

### Data parallelism behind one endpoint

A three-replica KVarN configuration provided one native endpoint with full
context and vision, but each request ran on one RTX 3090. Results were:

| Workload | DP=3 | TP=2 control | Change |
|---|---:|---:|---:|
| 4K/512, C1 output | 83.57 tok/s | 124.94 tok/s | -33.1% |
| 4K/512, C3 aggregate | 215.0 tok/s | 148.6 tok/s | +44.7% |
| Cold 8K prefill, one request | 1,077 tok/s | 1,566 tok/s | -31.2% |
| Cold 8K prefill, three requests | 2,933 tok/s | 1,589 tok/s | +84.6% |

The aggregate gains were real, but they came from sacrificing every individual
request. Prefix caches were replica-local and the available endpoint provided
no usable session affinity. KVarN also changes the KV precision boundary. The
configuration violated both the latency and cache contracts and was rejected.

### Partial and vision-only offload

Moving arbitrary language layers to GPU 0 would create a pipeline boundary over
the x8 PCIe link and has no supported placement path that preserves the current
DFlash execution. Moving only the vision tower could reduce image-encoding
latency, but would not improve text decode, text prefill or aggregate text
throughput. The current CPU-offloaded tower already preserves the full KV pool
and vision correctness. Neither option justified a custom runtime fork.

## Revisit conditions

Do not acquire a replacement board solely for this model: PCIe bandwidth does
not fix the TP=3 blocker. Reopen this campaign only if the host obtains either:

- x16 lanes for all three GPUs; or
- PCIe Gen 4 bandwidth across the participating slots.

and a new campaign also has at least one of:

- a target whose tensor-parallel dimensions are divisible by three;
- runtime support for uneven TP=3; or
- DFlash/speculative decoding support with PP=3.

Retain the same acceptance gates: one universal endpoint, full vision/context
and predictable cache reuse, with no important metric regressing more than 15%.
Given the present results, a candidate should target at least a 20-30% material
gain before its added complexity is considered worthwhile.

## Restoration

After the campaign, the TP=2/K7 production profile was restored. Health passed,
the API battery passed 12/12, the display manager remained active, and the power
caps remained 200/225/225 W on physical GPUs 0/1/2.
