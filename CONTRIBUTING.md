# Contributing to TrainKernel

TrainKernel is a research preview. Contributions should keep the supported
static PyTorch training subset explicit and preserve reproducible evidence.

Before opening a change:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Release changes must update `CHANGELOG.md` and `RELEASE_NOTES.md` before the
version tag is created.

CUDA changes should include a focused A100 parity check when an A100 is
available. Benchmark claims must include the workload, seed policy, precision,
baseline configuration, timing protocol, and limitations. Do not commit API
keys, SSH keys, pod identifiers, credentials, or generated build directories.

By contributing, you agree that your work is provided under the Apache License
2.0 in `LICENSE`.
