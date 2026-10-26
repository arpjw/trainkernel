# GPU infrastructure

## Initial validation pod (2026-10-25)

- GitHub private repository: `https://github.com/arpjw/megakernel`.
- RunPod access uses the official OAuth MCP integration at `https://mcp.runpod.io/`; no API key is stored in this project.
- One Secure Cloud validation pod was used for the parent compiler tests: `vbx2h07susuqdi`, A100 SXM4 80 GB, US-MD-1, quoted at `$1.79/hour` while running. It has now been terminated and verified absent.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; host driver CUDA 13.2, `nvcc` 12.8.93, PyTorch 2.8.0+cu128.
- SSH endpoint `ssh root@154.54.102.48 -p 10550` was used with the dedicated project key held outside the repository. SSH authentication and CUDA checks passed.
- The temporary project key was removed after validation. The two pre-existing account keys were preserved exactly.

## Lifecycle

The pod had no network volume and only exposed SSH. After compiler validation, only `vbx2h07susuqdi` was terminated, list/get reads confirmed it was absent, and only the temporary project public key was removed. No unrelated resources changed.

## Verification

Direct SSH checks passed on the pod: `nvidia-smi` reported the A100, `nvcc --version` reported CUDA 12.8.93, and PyTorch reported CUDA available with a successful CUDA tensor operation. The [initial A100 report](../results/a100/README.md) contains the compiler validation from that pod.

## Optimization pod (2026-10-26)

- Project-created pod `i49aq9zez15ek9` ran one A100 SXM4 80 GB in Secure Cloud US-MD-1. The quoted running rate was `$1.79/hour`. The image was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with a 50 GB ephemeral disk and no network volume.
- Access used a dedicated temporary SSH key outside this repository and Runpod OAuth tools. No Runpod API key was used or stored. Existing account keys and unrelated resources were not changed.
- The [optimization report](../results/a100-optimization/README.md) retains the twelve-run held-out matrix, 113-test A100 log, sanitizer results, generated CUDA, profiler record, and environment details. Results were copied to the repository workspace before teardown.
- Clock locking and hardware counters were denied by the pod's permissions, as recorded in the [clock attempt](../results/a100-optimization/evidence/clock-policy.json) and [counter attempt](../results/a100-optimization/evidence/ncu-tiny.csv). Timings therefore used unlocked clocks.
- After evidence was saved, only this project-created pod was terminated. Runpod returned HTTP 204 for deletion, and a follow-up get returned HTTP 404. The dedicated temporary SSH key was removed locally.
