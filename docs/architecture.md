# Architecture

## Hardware

The inference server is a Debian 13 host with an Intel i7-6900K, 94 GiB RAM and
three 24 GiB RTX 3090 cards. The durable inference target is the pair in physical
slots/buses `02:00.0` and `03:00.0`; both negotiate PCIe 3.0 x16 under load.
The card at `01:00.0` runs at x8 and is deliberately excluded because it is
expected to leave the host.

The excluded card was evaluated in TP, PP, DP and partial-offload designs.
No layout met the single-endpoint latency, context, vision, cache and speculation
contract. The measured results and revisit conditions are recorded in
[Three-GPU evaluation](three-gpu-evaluation.md).

There is no usable NVLink/P2P path between the cards. TP=2 therefore uses NCCL
through host PCIe. Despite that limitation, splitting the target across the two
x16 cards is faster: non-speculative decode rose from 34.4 tok/s on one card to
69.1 tok/s on two.

## Runtime

- Serving repository: `/home/ai/qwen-serving`
- Repository revision: `e3dc770483ab2f6e2305dd573b86af72a8e6df36`
- vLLM: `0.27.1+cu129`
- PyTorch: `2.13+cu129`
- Driver: `550.163.01`
- Stock comparison revision: `1f05c441c4e64ae0549de44fa9ea5a6d43610314`
- DFlash2 W4A16 revision: `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`
- Selected Huihui W4A16 revision: `c20530baefe3e77ccfc6891c2b50cce7ea28bf1e`

The target and DFlash2 drafter are W4A16 `compressed-tensors` checkpoints. The
optimized target also quantizes the large untied embedding/lm_head and MTP path,
which provides the VRAM margin for BF16 KV, CUDA graphs and the full context
contract.

At startup the selected profile reports roughly 11.4 GiB of KV cache per worker
and about 360K aggregate token capacity. This is sufficient for one request at
the native 262,144-token limit plus margin, or several shorter requests. It is
not a promise that four independent maximum-length requests can coexist.

## Vision

The optimized target retains the complete Qwen vision path: its checkpoint
index contains 333 vision weights, including patch embedding, 27 transformer
blocks and the merger/projector into the 5,120-wide language representation.
The selected profile enables one image per prompt, capped at 2,097,152 pixels
(2,048 image tokens).

On 24 GiB cards the approximately 0.85 GiB vision tower is CPU-offloaded and
copied module-by-module for image encoding. This preserves the measured
366,072-token final-launch KV pool and avoids a startup OOM during CUDA graph capture. The
trade-off is additional image-encoding latency over a resident tower; it does
not change the text model weights, KV dtype or decode path. A generated image
with a red left half and blue right half was processed through the live API and
returned exactly `RED, BLUE`.

## Cache contract

The server uses automatic prefix caching with the hybrid Mamba cache in `align`
mode. Stable system/project context must precede append-only conversation and
tool turns. Cache entries are in GPU memory, LRU-managed and lost at service
restart. OpenCode's cache-preserving compaction remains a separate client-side
requirement; enabling the server flag alone cannot preserve a compaction request
that rewrites the early prompt.

## Quality-impact classification

The following are quality-affecting and require the documented canaries:

- W4A16 target quantization and quantized embedding/lm_head/MTP weights;
- the target's Mamba state is declared FP32 but served in FP16 for capacity;
- Huihui refusal-direction ablation;
- INT8 activation/prefill paths, FP8 KV and lower-bit KV if enabled later.

TP=2, BF16 KV and verified speculative decoding are intended to preserve target
semantics, but the patched stack is still guarded by output tests. Sparse greedy
path drift was observed across execution modes: the DFlash2, MTP and spec-off
profiles did not hash identically on every prompt, although each mode was stable
on repeat. Do not treat speculative decoding as empirically bit-identical here.

INT8 activations and lower-bit KV were not selected. Prefix caching makes their
cold-prefill benefit less attractive than their measured or expected quality
cost.
