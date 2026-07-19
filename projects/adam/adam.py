import torch
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                g = p.grad.data
                state['t'] += 1

                # update first and second moment estimates
                state['m'] = group['beta1'] * state['m'] + (1 - group['beta1']) * g
                state['v'] = group['beta2'] * state['v'] + (1 - group['beta2']) * g ** 2

                # bias correction
                m_hat = state['m'] / (1 - group['beta1'] ** state['t'])
                v_hat = state['v'] / (1 - group['beta2'] ** state['t'])

                # parameter update
                p.data -= group['lr'] * m_hat / (v_hat.sqrt() + group['epsilon'])