#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json, torch
out={"cuda_available":torch.cuda.is_available()}
if out["cuda_available"]:
    out.update(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    out["sm80_or_newer"]=torch.cuda.get_device_capability() >= (8,0)
print(json.dumps(out, indent=2))
PY
