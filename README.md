# Qwen3.8-27B on 2× RTX 3090

## Outcome

The selected low-latency profile uses the two PCIe 3.0 x16 RTX 3090 cards,
vLLM 0.27.1, a W4A16 target, the W4A16 DFlash2 k=7 drafter, BF16 KV cache,
automatic prefix caching, TP=2, four request slots and the native 262,144-token
server limit.

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
