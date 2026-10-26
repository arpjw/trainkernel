# GPU infrastructure

## Current access check (2026-10-25)

- GitHub private repository: `https://github.com/arpjw/megakernel` (private, empty at the time of this check).
- A local SSH key is present at `~/.ssh/id_ed25519`.
- The key reaches `ssh.runpod.io`; a direct gateway login without a pod-specific username/endpoint receives `Permission denied (publickey)`. This confirms only network reachability, not the status of any existing pod.
- No `RUNPOD_*` environment variables or RunPod API key were found in the scoped local configuration/keychain checks.
- No existing RunPod pod endpoint was found in shell history or SSH config.

## Provisioning plan

Once a RunPod API key or an existing pod endpoint is available, provision one on-demand GPU only for bounded experiments:

1. Prefer an A100 80 GB if available at a lower hourly rate; use an H100 80 GB when Hopper-specific kernels are being evaluated.
2. Use a CUDA/PyTorch image with Python 3.11+, CUDA 12.x, and a persistent volume only if needed for benchmark artifacts.
3. Restrict access to the user's existing SSH public key. Never place API keys in this repository or benchmark logs.
4. Record pod ID, GPU type, region, image, driver/CUDA versions, and hourly price here after creation.
5. Stop or terminate only pods created for this project after results are copied to the repository; do not alter unrelated workloads.

## Connectivity smoke test

The gateway test used:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 ssh.runpod.io true
```

It reached the RunPod SSH service and failed authentication. Because this probe omitted a pod-specific username/endpoint, it is not an existing-pod connectivity test; it only confirms gateway reachability.

## Existing-session search

- No RunPod console tab was present in the available browser inventory. The visible Dia session contained unrelated YouTube tabs.
- Narrow rollout-memory search found a prior CPU-only RunPod campaign for another project. Its pod was stopped after evidence archival; no reusable endpoint or credential was retained.
- A second historical note explicitly records that RunPod was not configured or signed in and no pod was created.

