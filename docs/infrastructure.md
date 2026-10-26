# GPU infrastructure

## Initial validation pod (2026-10-25)

- Release repository: `https://github.com/arpjw/trainkernel`.
- RunPod access uses the official OAuth integration; no API key is stored in this project.
- One temporary Secure Cloud validation pod was used for the parent compiler tests: A100 SXM4 80 GB, US-MD-1, quoted at `$1.79/hour` while running. It has now been terminated and verified absent.
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; host driver CUDA 13.2, `nvcc` 12.8.93, PyTorch 2.8.0+cu128.
- SSH authentication used a dedicated temporary project key held outside the repository. The endpoint, key material, and account key inventory are intentionally omitted.

## Lifecycle

The pod had no network volume and only exposed SSH. After compiler validation, the temporary project pod was terminated and a follow-up read confirmed it was absent. No unrelated resources changed.

## Verification

Direct SSH checks passed on the pod: `nvidia-smi` reported the A100, `nvcc --version` reported CUDA 12.8.93, and PyTorch reported CUDA available with a successful CUDA tensor operation. The [initial A100 report](../results/a100/README.md) contains the compiler validation from that pod.

## Optimization pod (2026-10-26)

- A project-created temporary pod ran one A100 SXM4 80 GB in Secure Cloud US-MD-1. The quoted running rate was `$1.79/hour`. The image was `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, with a 50 GB ephemeral disk and no network volume.
- Access used a dedicated temporary SSH key outside this repository and Runpod OAuth tools. No Runpod API key was used or stored. Existing account keys and unrelated resources were not changed.
- The [optimization report](../results/a100-optimization/README.md) retains the twelve-run held-out matrix, 113-test A100 log, sanitizer results, generated CUDA, profiler record, and environment details. Results were copied to the repository workspace before teardown.
- Clock locking and hardware counters were denied by the pod's permissions, as recorded in the [clock attempt](../results/a100-optimization/evidence/clock-policy.json) and [counter attempt](../results/a100-optimization/evidence/ncu-tiny.csv). Timings therefore used unlocked clocks.
- After evidence was saved, only this project-created pod was terminated. Runpod returned a successful deletion response and a follow-up read confirmed it was absent. The dedicated temporary SSH key was removed locally.

## Independent release-validation pod (2026-10-26)

- A fresh Secure Cloud pod ran one A100 SXM4 80 GB in US-WA-1 at the quoted
  `$1.79/hour` rate. This was a different host and data center from the
  optimization run.
- It used the same official PyTorch 2.8 image, a 50 GB ephemeral disk, and no
  network or persistent volume. Source was transferred without GitHub
  credentials and validated at git revision `2ab10aac1e91bb69d6b86d3d76d3ada4d57a115f`.
- The [release record](../results/release-validation/README.md) contains the
  116-test GPU pass, 100-update example, three-seed classification-head result,
  package checks, environment, and raw benchmark data.
- After the evidence was copied, the pod was terminated. A fresh provider read
  returned `404 Not Found`. The temporary project SSH key was removed from the
  account, and a read-back confirmed that the two pre-existing keys were
  unchanged.
