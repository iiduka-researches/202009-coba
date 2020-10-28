from typing import Any, Callable, Dict
import torch


def get_cg_param_fn(cg_type: str) -> Callable:
    return _cg_param_dict[cg_type]


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def cg_param_hs(grad: torch.Tensor, deterministic_grad: torch.Tensor, deterministic_cg: torch.Tensor,
                group: Dict[str, Any]) -> torch.Tensor:
    y = grad - deterministic_grad
    dy = inner(deterministic_cg, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    return inner(grad, y) / (dy + eps)


def cg_param_fr(grad: torch.Tensor, deterministic_grad: torch.Tensor, deterministic_cg: torch.Tensor,
                group: Dict[str, Any]) -> torch.Tensor:
    deterministic_grad_norm_sq = inner(deterministic_grad, deterministic_grad)
    return inner(grad, deterministic_grad) / (deterministic_grad_norm_sq + group['eps'])


def cg_param_prp(grad: torch.Tensor, deterministic_grad: torch.Tensor, deterministic_cg: torch.Tensor,
                 group: Dict[str, Any]) -> torch.Tensor:
    y = grad - deterministic_grad
    deterministic_grad_norm_sq = inner(deterministic_grad, deterministic_grad)
    return inner(grad, y) / (deterministic_grad_norm_sq + group['eps'])


def cg_param_dy(grad: torch.Tensor, deterministic_grad: torch.Tensor, deterministic_cg: torch.Tensor,
                group: Dict[str, Any]) -> torch.Tensor:
    y = grad - deterministic_grad
    dy = inner(deterministic_cg, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    return inner(grad, deterministic_grad) / (dy + eps)


def cg_param_hz(grad: torch.Tensor, deterministic_grad: torch.Tensor, deterministic_cg: torch.Tensor,
                group: Dict[str, Any]) -> torch.Tensor:
    y = grad - deterministic_grad
    dy = inner(deterministic_cg, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    cg_param = inner(grad, y) / dy.add(eps)

    cg_param.add_(inner(y, y) * inner(grad, deterministic_cg) / (dy ** 2 + eps),
                  alpha=-group['lam'])
    _eta = torch.min(inner(grad, grad), torch.full_like(cg_param, group['eps']))
    eta = -1 / (inner(deterministic_cg, deterministic_cg) * _eta)
    return torch.max(cg_param, eta)


_cg_param_dict = dict(
    HS=cg_param_hs,
    FR=cg_param_fr,
    PRP=cg_param_prp,
    DY=cg_param_dy,
    HZ=cg_param_hz,
)
