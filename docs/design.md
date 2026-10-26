# Compiler design

## Capture

`capture()` flattens the input pytree and uses functionalization plus fake-mode FX tracing to obtain an ATen graph without applying updates to example inputs. PyTorch supplies automatic differentiation when the user's function contains `torch.func.grad_and_value`. The compiler lowers supported operations into a small SSA representation with explicit tensor shape, dtype, root buffer, offset, and strides.

The capture boundary rejects input writes, tensor constants hidden in closures, unsupported operators, and unsupported dtypes. Python scalar constants are specialized into the generated source. Views carry aliases instead of executing a new task. Materialized outputs use contiguous layouts; later view strides are derived from those layouts rather than blindly copied from tracing metadata.

## Scheduling and storage

The first scheduler processes operations in topological order. Single-consumer pointwise producers can be evaluated inline by their consumer, preserving rounding at each logical dtype boundary. An arena planner keeps roots alive through their last materialized reader, including readers reached through inline expressions and views. Returned values stay live until final output copies. Output tensors are independent contiguous materializations; alias identity and original view strides are outside the supported contract.

A cooperative grid remains resident for the whole step. All blocks execute the same operation sequence and stride over output tiles. The compiler groups independent operations into stages and inserts a barrier when a read-after-write, write-after-read, or write-after-write hazard crosses the current stage. It compares workspace intervals as well as logical dependencies, so arena reuse cannot silently create a race. A single-block grid uses block barriers. This remains a static schedule rather than a dependency-counter task queue; no spinning consumer can occupy the only resources on which its producer could run.

Short reductions with multiple outputs assign an output to each thread, keeping adjacent feature reads coalesced. Other reductions use block-local shared-memory reduction. The grid is capped by both occupancy and the graph's available tile work.

## Matrix multiplication

The BF16/FP16 path normally assigns four warps to a 32 by 32 output tile. Matrices with both output dimensions at most 16 use one active warp and a 16 by 16 tile to reduce padding and shared-memory traffic. Warps compute 16 by 16 accumulator fragments over K tiles of 16. The block stages aligned A/B tiles in shared memory. Tail elements are zero-padded. `addmm` applies its bias/alpha/beta epilogue before the output dtype conversion. FP32 uses a tiled SIMT path rather than silently reducing precision to TF32.

The host computes active blocks per SM for the compiled kernel and clamps the cooperative grid to that occupancy. The block size is fixed at 128 threads, matching four WMMA warps and the reduction implementation's power-of-two requirement. GPU capability and cooperative-launch support are checked.

## Runtime

CUDA source is compiled through PyTorch's extension builder and cached using a source, architecture, and PyTorch-version digest. Calls validate the static tensor signature and allocate independent output/workspace storage. `compiled.runner.allocate()` and `run_into()` expose caller-owned buffers for experiments, with device, layout, size, and alias checks. Shared execution buffers require the caller to order reuse correctly.

This compiler does not replace Python's training loop or provide autograd registration for an arbitrary compiled forward function. The caller compiles an explicit, pure training function returning gradients and updated state.

## Next experiments

The useful next optimizations depend on profiles: fuse matrix epilogues more broadly, use higher-throughput GEMM tiles, reduce global barriers through tile dependencies, store temporary BF16 values in native-width buffers, and add a cost model that can split the graph when separate library kernels are faster. Hopper-specific pipelines require their own measurements. None of these is a measured result in the initial implementation.
