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
| Stock DFlash2 k=7, 262K/cache-on baseline | 2 | 163.2 |
| **Selected Huihui, same runtime** | 2 | **167.2** |

The 262K/cache-on profile is selected. Its roughly 4% short-prompt cost versus
the stripped benchmark profile buys the native context and mandatory cache
contract.

The fixed 225 W p565/g512 power fixture exposes a workload-dependent reversal:

| Target and speculator | Decode tok/s |
|---|---:|
| Stock, DFlash2 k=7 | **180.3** |
| Huihui, DFlash2 k=7 | **142.5** |
| Huihui, native MTP4 | **137.5** |

DFlash2 is still the faster Huihui path and remains selected. The Huihui and
stock outputs differ, and speculative modes have shown sparse greedy hash drift,
so this is a real workload observation rather than a target-kernel-only A/B.
It is a performance caveat, not evidence of a quality regression. The planned
Huihui power sweep must measure decode and prefill separately.

| Target | Distinct 4K streams | Per-stream tok/s | Decode aggregate tok/s | Preemptions |
|---|---:|---:|---:|---:|
| Stock | 2 | 103.5 | 241.8 | 0 |
| Huihui | 2 | 112.1 | 269.7 | 0 |
| Stock | 4 | 57.6 | 309.8 | 0 |

Four users are viable, but the intended priority remains one user at maximum
speed. Two users are the best compromise when latency matters.

## Three-GPU campaign

The third, x8-attached RTX 3090 did not produce a worthwhile universal endpoint.
DP=3 improved three-request aggregate decode by 44.7% and aggregate 8K prefill
by 84.6%, but reduced the corresponding single-request scores by 33.1% and
31.2% and made cache reuse replica-dependent. PP=3 improved cold prefill but
cannot retain DFlash speculation. TP=3 is incompatible with the target's 32
attention heads.

A DFlash block sweep and attention-backend A/B also failed to provide a material
free gain. K3 delivered +5.8% at C4 for -6.1% C1 throughput and +10.8% C1 TPOT;
FlashInfer reduced C1 throughput by approximately 33%. Production remains
TP=2, DFlash2 k=7. Full fixtures, power measurements and the hardware threshold
for revisiting this work are in
[Three-GPU evaluation](three-gpu-evaluation.md).

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
GSM8K item out of 200. The 0.5-point GSM8K delta is inside sampling noise at
this sample size. It passed the functional, tool and cache gates and was 2.5%
faster, so it is the selected abliterated target; stock remains the rollback.

Other public abliterated Qwen3.8-27B families were screened before promotion.
Published results for OBLITERATUS V3, orcarouter, Jonathan/twolven, windowsxp
and hotdogs all contain larger capability losses on at least one reported
benchmark. Junafinity publishes no independent capability benchmark and no
ready equivalent Ampere W4A16 target. None provided a stronger zero-regression,
zero-throughput-loss candidate worth a full local qualification campaign.

These are bounded regression gates, not proof of general equivalence. Rerun the
same fixtures after model, vLLM, kernel, KV dtype, speculation or cache changes.
