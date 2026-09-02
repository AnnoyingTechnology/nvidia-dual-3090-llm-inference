# Benchmarks and quality

## Performance summary

All rows use the enforced 200 W/card cap. Short single-stream measurements use
approximately 512 input and 512 generated tokens. Concurrency rows use distinct
approximately 4K-token prefixes and 512 generated tokens.

| Runtime | GPUs | Single-stream tok/s |
|---|---:|---:|
| Speculation off | 1 | 34.4 |
| MTP4 | 1 | 96.3 |
| DFlash2 k=7 | 1 | 93.3 |
| Speculation off | 2 | 69.1 |
| MTP4 | 2 | 154.8 |
| DFlash2 k=7, stripped 65K/cache-off | 2 | 169.9 |
| DFlash2 k=7, 262K/cache-on candidate | 2 | 163.2 |
| Huihui candidate, same runtime | 2 | 167.2 |

The 262K/cache-on profile is selected. Its roughly 4% short-prompt cost versus
the stripped benchmark profile buys the native context and mandatory cache
contract.

| Distinct 4K streams | Per-stream tok/s | Decode aggregate tok/s | Preemptions |
|---:|---:|---:|---:|
| 2 | 103.5 | 241.8 | 0 |
| 4 | 57.6 | 309.8 | 0 |

Four users are viable, but the intended priority remains one user at maximum
speed. Two users are the best compromise when latency matters.

## Cache and long context

| Fixture | Cold | Warm/follow-up | Reused tokens | Result |
|---|---:|---:|---:|---|
| Stock, synthetic tool flow (~16K) | 11.27 s | 0.64 s | 15,680 | Exact |
| Stock, 100K needle at 90% depth | 86.67 s | 1.07 s | 99,456 | Exact |
| Huihui, synthetic tool flow (~16K) | 11.25 s | 0.63 s | 15,680 | Exact |
| Huihui, 30K needle at 90% depth | 21.64 s | 0.56 s | 29,568 | Exact |

The stock profile passed the full 12/12 API smoke battery, including greedy and
seeded determinism, log probabilities, structured JSON, streaming, thinking,
penalties and a 20K prompt. Huihui also passed 12/12.

## Quality comparison

| Target | English PPL | Danish PPL | Code PPL | Aggregate PPL | GSM8K |
|---|---:|---:|---:|---:|---:|
| Stock optimized W4A16 | 10.765 | 10.908 | 3.132 | 8.1355 | 96.0% |
| Huihui abliterated W4A16 | 10.769 | 10.934 | 3.156 | 8.1589 | 95.5% |

Huihui's aggregate perplexity is 0.29% worse and it misses one additional
GSM8K item out of 200. The difference is small, but it is a measured regression;
therefore it is optional rather than the default under a strict no-regression
rule.

These are bounded regression gates, not proof of general equivalence. Rerun the
same fixtures after model, vLLM, kernel, KV dtype, speculation or cache changes.
