"""Record the actual GPU kernel events for one compiled training step."""
import argparse
import json
import hashlib
from pathlib import Path
import torch
from benchmarks.train_mlp import make_inputs, step
from trainkernel import compile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    params, x, target = make_inputs(32, 128, 256, 64, torch.bfloat16, 'cuda')
    fn = lambda p, x, target: step(p, x, target, 1e-3)
    compiled = compile(fn, (params, x, target))
    compiled(params, x, target)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
        outputs = compiled(params, x, target)
        torch.cuda.synchronize()
    events = [event for event in profile.events() if 'CUDA' in str(event.device_type)]
    result = {'source_sha256': hashlib.sha256(compiled.source.encode()).hexdigest(), 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'gpu_events': [{'name': event.name, 'duration_us': event.device_time_total} for event in events],
              'megakernel_launches': sum('megakernel_entry' in event.name for event in events),
              'loss': float(outputs[0])}
    inductor = torch.compile(fn, mode="max-autotune", fullgraph=True)
    for _ in range(4):
        inductor(params, x, target)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as baseline_profile:
        baseline_outputs = inductor(params, x, target)
        torch.cuda.synchronize()
    baseline_events = baseline_profile.events()
    result['inductor_cuda_graph_launches'] = sum('cudaGraphLaunch' in event.name for event in baseline_events)
    result['inductor_gpu_events'] = [event.name for event in baseline_events if 'CUDA' in str(event.device_type)]
    result['inductor_loss'] = float(baseline_outputs[0])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.with_suffix('.cu').write_text(compiled.source)
    args.json.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    assert result['megakernel_launches'] == 1, 'expected exactly one generated GPU kernel'


if __name__ == '__main__':
    main()
