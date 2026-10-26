"""Validate the high-level training contract before compiling device code."""
import pytest
import torch
from trainkernel import CompiledTrainingStep, compile_training


def test_training_api_returns_typed_compiled_object(monkeypatch):
    parameter = torch.randn(3, 2)
    inputs = torch.randn(4, 3)

    class FakeRunner:
        source = "// generated CUDA"

        def __call__(self, parameter, inputs):
            loss = (inputs @ parameter).mean()
            return loss, torch.ones_like(parameter), parameter - 1e-3

    monkeypatch.setattr(
        "trainkernel.runtime.compile_program",
        lambda *args, **kwargs: FakeRunner(),
    )

    def step(parameter, inputs):
        value = (inputs @ parameter).mean()
        return value, torch.ones_like(parameter), parameter - 1e-3

    compiled = compile_training(step, (parameter, inputs))
    assert isinstance(compiled, CompiledTrainingStep)
    assert compiled.source == "// generated CUDA"
    actual = compiled(parameter, inputs)
    assert actual[0].shape == ()
    assert actual[1].shape == parameter.shape
    assert actual[2].shape == parameter.shape


def test_training_api_rejects_wrong_structure_before_cuda():
    p,x=torch.randn(3,2),torch.randn(4,3)
    with pytest.raises(ValueError,match='return loss, gradients'):
        compile_training(lambda p,x:x@p,(p,x))
    with pytest.raises(ValueError,match='pytree'):
        compile_training(lambda p,x:((x@p).mean(),(p,),p),(p,x))
    with pytest.raises(ValueError,match='scalar loss'):
        compile_training(lambda p,x:(x@p,p,p+1),(p,x))


def test_training_api_rejects_gradient_shape():
    p,x=torch.randn(3,2),torch.randn(4,3)
    with pytest.raises(ValueError,match='gradient.*shapes'):
        compile_training(lambda p,x:((x@p).mean(),p.sum(),p+1),(p,x))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_training_api_derives_nested_state_mapping():
    p={'weight':torch.randn(3,2,device='cuda'),'bias':torch.randn(2,device='cuda')}
    x=torch.randn(4,3,device='cuda')
    def step(p,x):
        def loss(q): return ((x@q['weight']+q['bias'])**2).mean()
        grads,value=torch.func.grad_and_value(loss)(p)
        return value,grads,{k:v-1e-3*grads[k] for k,v in p.items()}
    expected=step(p,x)
    compiled=compile_training(step,(p,x),max_blocks=1)
    assert isinstance(compiled,CompiledTrainingStep)
    actual=compiled(p,x)
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-5)
    assert actual[2]['weight'].data_ptr()==p['weight'].data_ptr()
    torch.testing.assert_close(p,expected[2],rtol=2e-4,atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_graph_factory_restores_state_and_replays_updates():
    from trainkernel import compile
    x=torch.randn(128,device='cuda')
    original=x.clone()
    compiled=compile(lambda x:x+1,(x,),inplace_outputs={0:0},max_blocks=1)
    captured=compiled.capture(x)
    torch.testing.assert_close(x,original)
    for _ in range(4):captured.replay()
    torch.testing.assert_close(x,original+4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_graph_factory_retains_inputs_after_container_edit():
    import gc
    import weakref
    from trainkernel import compile
    data={'x':torch.randn(128,device='cuda')}
    original=data['x'].clone()
    reference=weakref.ref(data['x'])
    compiled=compile(lambda data:data['x']+1,(data,),max_blocks=1)
    captured=compiled.capture(data)
    data['x']=torch.zeros_like(original)
    gc.collect()
    assert reference() is not None
    torch.testing.assert_close(captured.replay(),original+1)
