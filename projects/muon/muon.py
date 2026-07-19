import torch
from torch.optim import Optimizer


@torch.compile
def newton_schulz(M, steps=5):
    """Approximate the polar decomposition via Newton-Schulz iteration."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M / (M.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            ns_steps = group['ns_steps']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                buf = state['momentum_buffer']
                buf.mul_(mu).add_(grad)

                if nesterov:
                    update = grad + mu * buf
                else:
                    update = buf.clone()

                transposed = False
                if update.shape[0] < update.shape[1]:
                    update = update.T
                    transposed = True

                update = newton_schulz(update, steps=ns_steps)

                if transposed:
                    update = update.T

                p.data.add_(update, alpha=-lr)