import pytest
import torch

from benchmarks.train_mlp import compare, make_inputs, step, trajectory
from megakernel import compile as mk_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SM80+")


@pytest.fixture(autouse=True)
def accurate_matmul():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("requires CUDA SM80+")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_compiler_training_step_and_trajectory(dtype):
    params, x, target = make_inputs(17, 33, 65, 19, dtype, torch.device("cuda"))
    fn = lambda p, xx, yy: step(p, xx, yy, 1e-3)
    compiled = mk_compile(fn, (params, x, target))
    compare(compiled(params, x, target), fn(params, x, target), dtype)
    trajectory(compiled, params, x, target, 5, dtype)
    # Compile-time signatures must be enforced before passing raw pointers.
    with pytest.raises(ValueError, match="shape, stride"):
        compiled(params, x[:1], target)


def test_graph_replay_and_execution_buffer_guards():
    x = torch.randn(17, 33, device="cuda")
    compiled = mk_compile(lambda a: (a * 2 + 1).mean(dim=0, keepdim=True), (x,))
    expected = (x * 2 + 1).mean(dim=0, keepdim=True)
    compiled(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = compiled(x)
    graph.replay()
    torch.testing.assert_close(output, expected)
    buffers = compiled.runner.allocate()
    buffers.workspace = x.view(-1)
    with pytest.raises(ValueError, match="alias|workspace"):
        compiled.runner.run_into(x, buffers=buffers)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_transpose_broadcast_partial_reduction(dtype):
    x = torch.randn(5, 7, 3, device="cuda", dtype=dtype).transpose(0, 2)
    bias = torch.randn(5, device="cuda", dtype=dtype)
    fn = lambda a, b: (a + b).sum((0, 2), keepdim=True)
    compiled = mk_compile(fn, (x, bias))
    torch.testing.assert_close(compiled(x, bias), fn(x, bias), rtol=0.02, atol=0.01)
