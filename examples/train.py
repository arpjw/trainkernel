"""Compile an nn.Module's functional loss, gradients, and SGD updates together."""
import torch
import megakernel


def main():
    device = "cuda"
    model = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.ReLU(), torch.nn.Linear(256, 64)).to(device)
    params = dict(model.named_parameters())
    x = torch.randn(32, 128, device=device)
    target = torch.randn(32, 64, device=device)

    def train_step(p, inputs, labels):
        def loss(p):
            prediction = torch.func.functional_call(model, p, (inputs,))
            return ((prediction - labels) ** 2).mean()
        gradients, value = torch.func.grad_and_value(loss)(p)
        updated = {name: weight - 1e-3 * gradients[name] for name, weight in p.items()}
        return value, gradients, updated

    compiled = megakernel.compile(train_step, (params, x, target))
    for iteration in range(10):
        loss, gradients, params = compiled(params, x, target)
        print(f"step={iteration} loss={loss.item():.6f}")
    print(f"one CUDA launch; {compiled.runner.generated.workspace_elements * 4} workspace bytes")


if __name__ == "__main__":
    main()
