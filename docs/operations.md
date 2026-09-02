# Operations

## Current state

The validated service listens on the inference host's LAN address:

```text
http://<inference-host>:18020/v1
```

The API key is stored at `/home/ai/qwen-serving/api_key.txt` with mode 0600.
Do not copy it into this repository, a shell history or logs. The current server
is a manually launched validation process; no persistent vLLM systemd unit has
been installed yet. The OpenAI-compatible model ID is `qwen3.8-27b`. Requests
without the bearer token are rejected with HTTP 401.

## Start the selected profile

Run on the inference host as the service account:

```bash
cd /home/ai/qwen-serving
CUDA_VISIBLE_DEVICES=1,2 \
MODEL=/home/ai/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound-fast \
CTX=fast SPEC=dflash2 PREFIX_CACHE=1 VISION=1 VISION_OFFLOAD=1 \
DFLASH_MAX_LEN=262144 MAX_SEQS=4 \
HOST=0.0.0.0 PORT=18020 EXTRA_ARGS="--tensor-parallel-size 2" \
single-user/start_qwen.sh
```

Use the Huihui model path from `profiles/huihui.env` only for the optional
abliterated profile. All other runtime settings must remain identical for a
meaningful A/B.

## Validate

On the inference host, with the service healthy:

```bash
cd /home/ai/qwen-serving
venv/bin/python bench/api_smoke.py
```

The repository scripts are safe synthetic canaries. Run them by copying the
script to the host or piping it to `/home/ai/qwen-serving/venv/bin/python -`.
Acceptance requires:

- 12/12 API smoke tests;
- correct parsed tool call and exact tool-result answer;
- nonzero `cached_tokens` on the append-only follow-up;
- exact cold/warm needle retrieval;
- exact image interpretation through `scripts/vision_canary.py`;
- no material single-stream or 2-user throughput regression.

## Prefix-cache behavior

Cache reuse depends on an identical rendered prefix, including system prompts,
tool schemas and early messages. Append tool turns; do not rewrite prior turns.
Changing early content, changing tool definitions or restarting vLLM invalidates
reuse. The cache is an acceleration mechanism, not durable storage.

The cache-preserving OpenCode compaction fork documented in the B70 reference
repository is compatible with this OpenAI endpoint. Its opt-in client behavior
must remain enabled when switching providers; this server cannot recover reuse
after the client changes the early prompt.

## Power policy

The host applies the original 200 W all-GPU limit first, then raises only the
two x16 inference cards to 225 W. The example systemd drop-in shipped under
`systemd/` uses placeholders: replace them with the two local GPU UUIDs before
installation. UUID selection makes the policy independent of index changes
when the x8 card is removed.

Rollback is removal of
`/etc/systemd/system/nvidia-power-limit.service.d/override.conf`, followed by
`systemctl daemon-reload` and a restart of `nvidia-power-limit.service`. This
restores the original 200 W/card policy. Always verify with:

```bash
nvidia-smi -i 1,2 --query-gpu=index,enforced.power.limit --format=csv,noheader
```

## Headless conversion

The host boots `graphical.target`, runs GDM auto-login, GNOME Shell and GNOME
Remote Desktop. The live desktop uses roughly 1.2–1.5 GiB RAM and 258 MiB VRAM
on the excluded x8 GPU. Converting to headless operation mainly reduces
background activity and administrative surface; it does not directly free VRAM
on GPUs 1 and 2.

The safe change is:

1. set `multi-user.target` as the default;
2. disable and stop GDM after confirming SSH access;
3. purge explicit GNOME and GUI applications only;
4. do not purge `xserver-xorg-video-nvidia` or blanket-autoremove dependencies;
5. verify SSH, NetworkManager, NVIDIA modules, `nvidia-smi`, vLLM health and the
   200/225 W policy.

Rollback is reinstalling `gnome-core gdm3`, re-enabling GDM and restoring
`graphical.target`. Package removal requires a reviewed simulation immediately
before execution because Debian dependency state can change.
