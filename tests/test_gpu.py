import pytest
import torch

from benchmarks.train_mlp import compare, make_inputs, step, trajectory
from trainkernel import compile as mk_compile

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


def test_view_only_outputs_and_casts():
    x = torch.randn(3, 7, device="cuda")
    fn = lambda a: {"transpose": a.t(), "alias": a.detach(), "cast": torch.add(a, a, alpha=0.25).to(torch.bfloat16)}
    compiled = mk_compile(fn, (x,))
    actual, expected = compiled(x), fn(x)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


def test_nn_linear_training_addmm_epilogue():
    model = torch.nn.Sequential(torch.nn.Linear(11, 17), torch.nn.ReLU(), torch.nn.Linear(17, 5)).cuda()
    params = dict(model.named_parameters())
    x, target = torch.randn(7, 11, device="cuda"), torch.randn(7, 5, device="cuda")
    def fn(p, x, target):
        def loss(p):
            return ((torch.func.functional_call(model, p, (x,)) - target) ** 2).mean()
        gradients, value = torch.func.grad_and_value(loss)(p)
        return value, gradients, {key: p[key] - 1e-3 * gradients[key] for key in p}
    compiled = mk_compile(fn, (params, x, target))
    compare(compiled(params, x, target), fn(params, x, target), torch.float32)


def test_addmm_beta_zero_ignores_nan_bias():
    a = torch.randn(3, 7, device="cuda")
    b = torch.randn(7, 5, device="cuda")
    bias = torch.full((5,), float("nan"), device="cuda")
    fn = lambda a, b, c: (torch.addmm(c, a, b, beta=0), torch.addmm(c, a, b, alpha=0, beta=0))
    compiled = mk_compile(fn, (a, b, bias))
    for actual, expected in zip(compiled(a, b, bias), fn(a, b, bias)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_small_tensor_core_tile(dtype):
    a = torch.randn(3, 8, device='cuda', dtype=dtype)
    b = torch.randn(8, 5, device='cuda', dtype=dtype)
    compiled = mk_compile(lambda a, b: a @ b, (a, b))
    torch.testing.assert_close(compiled(a, b), a @ b, rtol=0.02, atol=0.004)
    assert compiled.runner.generated.max_work_blocks == 1
