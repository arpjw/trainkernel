# GPU infrastructure

## Current access (2026-10-25)

- GitHub private repository: `https://github.com/arpjw/megakernel`.
- RunPod access uses the official OAuth MCP integration at `https://mcp.runpod.io/`; no API key is stored in this project.
- One Secure Cloud validation pod was used for the parent compiler tests: `vbx2h07susuqdi`, A100 SXM4 80 GB, US-MD-1, quoted at `$1.79/hour` while running. It has now been terminated and verified absent.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; host driver CUDA 13.2, `nvcc` 12.8.93, PyTorch 2.8.0+cu128.
- SSH endpoint `ssh root@154.54.102.48 -p 10550` was used with the dedicated project key held outside the repository. SSH authentication and CUDA checks passed.
- The temporary project key was removed after validation. The two pre-existing account keys were preserved exactly.

## Lifecycle

The pod had no network volume and only exposed SSH. After compiler validation, only `vbx2h07susuqdi` was terminated, list/get reads confirmed it was absent, and only the temporary project public key was removed. No unrelated resources changed.

## Verification

Direct SSH checks passed on the active pod: `nvidia-smi` reported the A100, `nvcc --version` reported CUDA 12.8.93, and PyTorch reported CUDA available with a successful CUDA tensor operation. Full provisioning and validation evidence is in `work/runpod-pod.jsonl`.
