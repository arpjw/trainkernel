"""A complete A100 online-regression loop with FP32 state and BF16 compute.

Run from the repository with a CUDA development toolkit installed:
    python -m examples.online_training

The data is synthetic and resident on the GPU. This demonstrates the supported
training contract; it is not an application-level throughput benchmark.
"""
from __future__ import annotations
import math
import torch
import trainkernel


def training_step(state, x, target):
    def loss_fn(p):
        q = tuple(v.to(torch.bfloat16) for v in p)
        prediction = torch.relu(x @ q[0] + q[1]) @ q[2] + q[3]
        return ((prediction.float() - target.float()) ** 2).mean()
    gradients, loss = torch.func.grad_and_value(loss_fn)(state)
    return loss, gradients, tuple(p - 1e-3 * g for p, g in zip(state, gradients))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This example needs an SM80+ CUDA GPU and nvcc')
    torch.manual_seed(101)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    device = 'cuda'
    state = (torch.randn(32,64,device=device)/math.sqrt(32), torch.zeros(64,device=device),
             torch.randn(64,16,device=device)/math.sqrt(64), torch.zeros(16,device=device))
    initial = tuple(p.clone() for p in state)
    x = torch.randn(4,32,device=device,dtype=torch.bfloat16)
    target = torch.randn(4,16,device=device,dtype=torch.bfloat16)
    compiled = trainkernel.compile_training(training_step, (state,x,target), max_blocks=1)
    # capture() warms up, restores state, and binds the graph to these buffers.
    captured = compiled.capture(state, x, target, warmup=3)
    reference = initial
    for _ in range(100):
        loss, gradients, updated = captured.replay()
        expected = training_step(reference,x,target)
        reference = expected[2]
    torch.cuda.synchronize()
    torch.testing.assert_close(loss,expected[0],rtol=2e-4,atol=2e-5)
    torch.testing.assert_close(gradients,expected[1],rtol=0.04,atol=0.004)
    torch.testing.assert_close(state,reference,rtol=2e-4,atol=2e-5)
    print(f'100 stateful updates match eager. Final loss: {float(loss):.6f}')
    print('Each replay runs one generated CUDA training kernel.')

if __name__ == '__main__':
    main()
