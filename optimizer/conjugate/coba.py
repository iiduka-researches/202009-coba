import math

import torch

from torch.optim.optimizer import Optimizer
from optimizer.conjugate.conjugate_param2 import get_cg_param_fn


class CoBA(Optimizer):
    def __init__(self, params, period: int, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False,
                 cg_type='HS', lam=2.0, m=1e-3, a=1+1e-8) -> None:
        if not 0 <= period:
            raise ValueError("Invalid period: {}".format(period))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 < m:
            raise ValueError("Invalid m value: {}".format(m))
        if not 1.0 <= a:
            raise ValueError("Invalid a value: {}".format(a))
        defaults = dict(period=period, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        cg_type=cg_type, lam=lam, a=a, m=m)
        super(CoBA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CoBA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('cg_type', 'HS')

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            cg_param_fn = get_cg_param_fn(group['cg_type'])
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                    state['stochastic_cg'] = None
                    state['past_grad'] = None

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                else:
                    max_exp_avg_sq = None
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    # Decay the first and second moment running average coefficient
                    grad = grad.add(p, alpha=group['weight_decay'])

                if state['stochastic_cg'] is None:
                    state['past_grad'] = grad.clone()
                    state['stochastic_cg'] = (-grad).clone()
                else:
                    scg = state['stochastic_cg']
                    cg_param = cg_param_fn(grad, state['past_grad'], scg, group)
                    state['stochastic_cg'] = -grad + group['m'] * cg_param * scg / (state['step'] ** group['a'])
                    state['past_grad'] = grad.clone()

                exp_avg.mul_(beta1).add_(state['stochastic_cg'], alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    # denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    step_size = group['lr']
                else:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=step_size)

        return loss
