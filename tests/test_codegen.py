import pytest
import torch

from benchmarks.train_mlp import make_inputs, step
from megakernel import capture
from megakernel.codegen import _Generator, generate, numel


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_training_codegen_and_arena_lifetimes(dtype):
    params, x, y = make_inputs(7, 11, 17, 5, dtype, "cpu")
    program, _ = capture(lambda p, x, y: step(p, x, y, 1e-3), (params, x, y))
    scheduler = _Generator(program, blocks_per_sm=1, fuse_pointwise=True)
    generated = scheduler.generate()
    # Verify that no live operand or output storage is overwritten by another
    # allocation, including operands reached through fused pointwise expressions.
    live = set()
    produced = set()
    for i, op in enumerate(scheduler.materialized):
        def roots(spec):
            if spec.buffer not in scheduler.inline:
                return {spec.buffer}
            return set().union(*(roots(x) for x in scheduler.producers[spec.buffer].inputs if hasattr(x, "buffer")))
        live = set().union(*(roots(x) for future in scheduler.materialized[i:] for x in future.inputs if hasattr(x, "buffer")))
        live.update(output.buffer for output in program.outputs)
        start = scheduler.offsets[op.output.buffer]
        end = start + numel(op.output.shape)
        for root in produced.intersection(live):
            other_start = scheduler.offsets[root]
            other_end = other_start + numel(scheduler.producers[root].output.shape)
            assert end <= other_start or other_end <= start
        produced.add(op.output.buffer)
    assert generated.inlined_operations > 0
    assert generated.tensor_core_operations == (0 if dtype == torch.float32 else 5)
    assert generated.workspace_elements < sum(numel(op.output.shape) + 64 for op in program.operations)
    assert "cudaLaunchCooperativeKernel" in generated.source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in generated.source


def test_cpu_compile_has_explicit_error():
    from megakernel import compile
    with pytest.raises(RuntimeError, match="CUDA SM80"):
        compile(lambda x: x + 1, (torch.ones(4),))
