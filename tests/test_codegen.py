import pytest
import torch

from benchmarks.train_mlp import make_inputs, step
from trainkernel import capture
from trainkernel.codegen import _Generator, generate, numel


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
    # Three small-K GEMMs use the direct FP32 accumulator path; the remaining
    # two low-precision GEMMs use WMMA.
    assert generated.tensor_core_operations == (0 if dtype == torch.float32 else 2)
    assert generated.workspace_elements < sum(numel(op.output.shape) + 64 for op in program.operations)
    assert "cudaLaunchCooperativeKernel" in generated.source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in generated.source


def test_cpu_compile_has_explicit_error():
    from trainkernel import compile
    with pytest.raises(RuntimeError, match="CUDA SM80"):
        compile(lambda x: x + 1, (torch.ones(4),))


def test_independent_outputs_share_a_barrier_but_dependencies_do_not():
    x = torch.randn(33)
    independent, _ = capture(lambda x: (x + 1, x + 2), (x,))
    dependent, _ = capture(lambda x: (x + 1, (x + 1) * 2), (x,))
    # Independent direct outputs need no grid fence before kernel completion.
    assert generate(independent).grid_barriers == 0
    # Explicitly return a producer and its consumer to prevent inlining.
    def chain(x):
        y = x + 1
        return y, y * 2
    dependent, _ = capture(chain, (x,))
    assert generate(dependent).grid_barriers == 1


def test_gemm_epilogue_fusion_is_selective_and_optional():
    params, x, y = make_inputs(7, 11, 17, 5, torch.bfloat16, "cpu")
    program, _ = capture(lambda p, x, y: step(p, x, y, 1e-3), (params, x, y))
    fused = generate(program)
    unfused = generate(program, fuse_epilogues=False)
    assert fused.materialized_operations < unfused.materialized_operations
    assert fused.grid_barriers < unfused.grid_barriers
    # Disabling the pass preserves the baseline scheduler for bisecting GPU
    # numerical or performance changes.
    assert unfused.materialized_operations == 16
    assert unfused.grid_barriers == 8
