import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from benchmarks.train_mlp import make_inputs, step
from trainkernel import capture
from trainkernel.reference import run


def check(fn, args, **tolerances):
    program, output_spec = capture(fn, args)
    expected = fn(*args)
    flat_inputs, _ = tree_flatten(args)
    actual = tree_unflatten(list(run(program, flat_inputs)), output_spec)
    actual_flat, actual_tree = tree_flatten(actual)
    expected_flat, expected_tree = tree_flatten(expected)
    assert actual_tree == expected_tree
    for got, want in zip(actual_flat, expected_flat):
        torch.testing.assert_close(got, want, **tolerances)
    return program


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_full_training_step_and_trajectory(dtype):
    params, x, target = make_inputs(7, 11, 17, 5, dtype, "cpu")
    fn = lambda p, xx, yy: step(p, xx, yy, 1e-3)
    for _ in range(3):
        program = check(fn, (params, x, target))
        flat = run(program, (*params, x, target))
        params = tuple(flat[-4:])
    assert sum(op.op == "mm" for op in program.operations) == 5
    assert any(op.op == "threshold_backward" for op in program.operations)
    assert len(program.outputs) == 9


@pytest.mark.parametrize("dims", [(0,), (1,), (-1,), (0, 2), ()])
@pytest.mark.parametrize("keepdim", [False, True])
def test_partial_reductions(dims, keepdim):
    check(lambda x: {"sum": x.sum(dims, keepdim=keepdim), "mean": x.mean(dims, keepdim=keepdim)}, (torch.randn(2, 3, 4),))


def test_aliases_broadcast_and_transposed_inputs():
    x = torch.randn(3, 4).t()
    y = torch.randn(3)
    check(lambda a, b: (a.t(), a + b, a.unsqueeze(0).expand(2, 4, 3), a.contiguous().view(-1)), (x, y))


def test_alpha_cast_and_pytree():
    args = ((torch.randn(3, 4), torch.randn(3, 4)),)
    check(lambda p: {"value": torch.add(p[0], p[1], alpha=0.25).to(torch.bfloat16), "stats": (p[0].sum(),)}, args)


def test_input_mutation_is_rejected_without_mutating_examples():
    x = torch.ones(4)
    with pytest.raises(ValueError, match="mutation"):
        capture(lambda x: x.add_(2), (x,))
    torch.testing.assert_close(x, torch.ones(4))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float64])
def test_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="unsupported dtype"):
        capture(lambda x: x + 1, (torch.ones(3, dtype=dtype),))


def test_unsupported_operator_has_diagnostic():
    with pytest.raises(ValueError, match="unsupported ATen op aten.sin"):
        capture(torch.sin, (torch.ones(3),))


def test_nn_module_training_via_functional_call():
    model = torch.nn.Sequential(torch.nn.Linear(11, 17), torch.nn.ReLU(), torch.nn.Linear(17, 5))
    params = dict(model.named_parameters())
    x, y = torch.randn(7, 11), torch.randn(7, 5)
    def train(p, x, target):
        def loss(p):
            return ((torch.func.functional_call(model, p, (x,)) - target) ** 2).mean()
        grads, value = torch.func.grad_and_value(loss)(p)
        return value, grads, {key: p[key] - 1e-3 * grads[key] for key in p}
    program = check(train, (params, x, y))
    assert any(op.op == "addmm" for op in program.operations)
    from trainkernel.codegen import generate
    assert generate(program).materialized_operations > 0
