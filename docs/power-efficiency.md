# Power efficiency

## Scope

This campaign used the stock optimized target and integrates NVIDIA board-power
telemetry for physical GPUs 1 and 2.
It excludes CPU, RAM, storage, PSU losses and the unused third GPU, so the unit
is output tokens per GPU-board joule, not whole-system tokens per joule.

Every request used identical text and a fresh `cache_salt`, so the prefix cache
could not hide prefill. Output hashes were identical at every cap. Each point is
the median of three decode runs or two prefill runs. The original 200 W/card cap
was restored automatically after each campaign.

## Decode: 565 input, 512 output

| Cap per card | Decode tok/s | Total board W | tok/GPU-J |
|---:|---:|---:|---:|
| 125 W | 53.4 | 249.5 | 0.214 |
| 150 W | 102.9 | 298.1 | 0.348 |
| 175 W | 154.9 | 345.4 | **0.449** |
| 200 W | 172.7 | 390.2 | 0.443 |
| 225 W | 180.3 | 434.5 | 0.415 |
| 250 W | 184.3 | 478.7 | 0.385 |
| 275 W | 186.9 | 523.8 | 0.357 |
| 300 W | 188.1 | 566.4 | 0.332 |
| 350 W | 189.0 | 601.0 | 0.314 |
| 400 W | 189.0 | 602.6 | 0.313 |

Decode reaches its best measured energy efficiency at 175 W/card. Moving to
200 W adds 11.5% throughput for only a 1.3% efficiency loss. Moving from 200 to
225 W adds another 4.4% throughput but costs 6.3% efficiency. Above 225 W the
curve is flat.

## Prefill: 8,221 input, 8 output

| Cap per card | Prefill tok/s | Total board W | tok/GPU-J |
|---:|---:|---:|---:|
| 150 W | 966.9 | 285.0 | 3.400 |
| 175 W | 1,349.2 | 340.2 | **3.966** |
| 200 W | 1,499.0 | 382.2 | 3.924 |
| 225 W | 1,570.6 | 425.5 | 3.692 |
| 250 W | 1,618.1 | 469.3 | 3.449 |
| 300 W | 1,678.4 | 550.7 | 3.050 |

Prefill has the same useful region. At 200 W it gains 11.1% over 175 W with
only about 1.1% lower energy efficiency. At 225 W it gains 4.8% over 200 W for
about 6% lower efficiency.

## Decision

- **200 W/card:** energy-efficient knee; the rational default for a cache-heavy
  service.
- **225 W/card:** latency-biased knee; recommended for the stated maximum
  single-user preference.
- **Above 225 W/card:** reject for normal service. The extra heat and energy buy
  little throughput.

The persistent systemd policy is set to 225 W on the two inference cards and
200 W on the excluded x8 card. This is the approved interim cap for Huihui. A
separate final-target sweep remains pending because its output and speculative
acceptance differ from stock; decode and prefill must be measured independently.
Dynamic per-phase caps are not justified unless that sweep finds materially
different knees.
