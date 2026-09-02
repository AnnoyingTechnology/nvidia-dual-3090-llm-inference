# Qwen3.8-27B on 2× RTX 3090

## Outcome

The selected low-latency profile uses the two PCIe 3.0 x16 RTX 3090 cards,
vLLM 0.27.1, a W4A16 target, the W4A16 DFlash2 k=7 drafter, BF16 KV cache,
automatic prefix caching, CPU-offloaded vision, TP=2, four request slots and
the native 262,144-token server limit.

At the measured 200 W/card baseline it sustains about **163 tok/s** on the
repeated single-stream fixture. Two distinct 4K streams produce **242 aggregate
tok/s** and four produce **310 aggregate tok/s**, without preemption. The live
latency-biased policy is 225 W/card on the inference pair. The stripped
65K/cache-off tuning profile reached 170 tok/s, but is not the operational
candidate because full context and cache reuse are requirements.

Automatic prefix caching is mandatory. A cache-isolated 100,040-token retrieval
request took 86.7 seconds cold, then 1.07 seconds with 99,456 cached tokens and
the same exact answer. A synthetic OpenCode-compatible tool flow reduced 11.27
seconds cold to 0.64 seconds on the append-only follow-up while reusing 15,680
tokens.

## Measured result

Unless noted otherwise, these measurements use the selected 262K/cache-on
profile at the original 200 W/card cap. The live service now uses 225 W/card.

| Measurement | Result |
|---|---:|
| Single stream, approximately p512/g512 | **163.2 tok/s** |
| Two distinct 4K streams | **241.8 aggregate tok/s** |
| Four distinct 4K streams | **309.8 aggregate tok/s** |
| Decode power cell at 225 W, p565/g512 | **180.3 tok/s** |
| Cold prefill power cell at 225 W, p8,221/g8 | **1,570.6 tok/s** |
| Cold 100,040-token retrieval | **86.67 s** |
| Repeated 100K retrieval | **1.07 s**, 99,456 cached tokens |
| OpenCode-style tool follow-up | **0.64 s**, 15,680 cached tokens |
| Synthetic vision canary | Exact `RED, BLUE` |

The concurrency cells completed without scheduler preemption. Four users are
viable, while one user remains the latency priority and two users are the best
interactive compromise.

## Optimization ladder

| Runtime decision | Single-stream result | Boundary |
|---|---:|---|
| One x16 RTX 3090, speculation off | 34.4 tok/s | W4A16 target |
| Two x16 RTX 3090, TP=2, speculation off | 69.1 tok/s | Same target and fixture |
| TP=2 with MTP4 | 154.8 tok/s | Target-verified speculation |
| TP=2 with DFlash2 k=7, stripped 65K/cache-off | 169.9 tok/s | Tuning ceiling, not deployable contract |
| TP=2 with DFlash2 k=7, 262K/cache-on | **163.2 tok/s** | Selected production profile |

TP=2 remains worthwhile despite the lack of GPU P2P/NVLink: host PCIe
communication is cheaper than leaving half of the target bandwidth unused.
The production profile deliberately gives up about 4% against the stripped
tuning ceiling to retain native context and automatic prefix caching.

## Power curves

Decode and prefill were swept separately because they do not necessarily share
a knee. They converged on the same useful range here. Values are medians; output
hashes remained identical at every cap.

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

The best board-energy efficiency is 175 W/card. The 200 W point retains nearly
all of that efficiency while adding roughly 11% throughput. The applied 225 W
latency-biased knee adds another 4–5% for about 11% more board power. Above it,
returns collapse.

## Recommended profile

| Setting | Value |
|---|---|
| Physical GPUs | 1 and 2, PCIe 3.0 x16 |
| Tensor parallelism | 2 |
| Target | `Qwen3.8-27B-W4A16-AutoRound-fast` |
| Speculator | DFlash2 W4A16, 7 draft tokens |
| Context | 262,144 tokens |
| KV cache | BF16 |
| Prefix cache | Enabled, Mamba `align` mode |
| Vision | Enabled, one image/prompt, CPU-offloaded tower |
| Scheduler | 2,048 batched tokens, 4 sequences, async |
| API | OpenAI-compatible, `0.0.0.0:18020`, API-key protected |
| Model ID | `qwen3.8-27b` |
| Power | 225 W/card on the x16 pair; 200 W on the excluded x8 card |

The power campaign found the large performance gains exhausted by 200 W/card.
The applied 225 W setting adds about 4–5% to both decode and prefill for about
11% more GPU-board power. Above 225 W, the return is small; 300–400 W only
reaches about 189 tok/s on the fixed decode fixture.

## Abliterated candidate

`ababaka/Huihui-Qwen3.8-27B-Abliterated-W4A16-AutoRound` was tested with the
same runtime. It did not regress serving or cache behavior and was slightly
faster on the sampled prompts, but aggregate perplexity increased 0.29% and
GSM8K moved from 96.0% to 95.5% (one additional miss out of 200). It is retained
as an optional uncensored profile, not promoted as the default under the stated
no-regression rule.

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmarks and quality](docs/benchmarks-and-quality.md)
- [Power efficiency](docs/power-efficiency.md)
- [Operations](docs/operations.md)
- [References](docs/references.md)

Raw JSON evidence is under `results/`. Model weights, virtual environments,
logs and API credentials are intentionally excluded.
