from typing import Any, Callable, Dict
import torch


def get_cg_param_fn(cg_type: str) -> Callable:
    return _cg_param_dict[cg_type]


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def cg_param_hs(grad: torch.Tensor, g_buf: torch.Tensor, d_buf: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
    y = grad - g_buf
    dy = inner(d_buf, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    # dy[(dy >= 0) & (dy < group['eps'])] = group['eps']
    # dy[(dy < 0) & (dy > -group['eps'])] = -group['eps']
    # return inner(grad, y) / dy
    return inner(grad, y) / (dy + eps)


def cg_param_fr(grad: torch.Tensor, g_buf: torch.Tensor, d_buf: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
    g_buf_norm_sq = inner(g_buf, g_buf)
    return inner(grad, grad) / (g_buf_norm_sq + group['eps'])


def cg_param_prp(grad: torch.Tensor, g_buf: torch.Tensor, d_buf: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
    y = grad - g_buf
    g_buf_norm_sq = inner(g_buf, g_buf)
    return inner(grad, y) / (g_buf_norm_sq + group['eps'])


def cg_param_dy(grad: torch.Tensor, g_buf: torch.Tensor, d_buf: torch.Tensor, group: Dict[str, Any]) -> torch.Tensor:
    y = grad - g_buf
    dy = inner(d_buf, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    # dy[(dy >= 0) & (dy < group['eps'])] = group['eps']
    # dy[(dy < 0) & (dy > -group['eps'])] = -group['eps']
    return inner(grad, grad) / (dy + eps)


def cg_param_hz(grad: torch.Tensor, g_buf: torch.Tensor, d_buf: torch.Tensor,
                group: Dict[str, Any]) -> torch.Tensor:
    y = grad - g_buf
    dy = inner(d_buf, y)
    eps = torch.full_like(dy, group['eps'])
    eps = torch.where(dy >= 0, eps, -eps)
    cg_param = inner(grad, y) / dy.add(eps)
    # dy[(dy >= 0) & (dy < group['eps'])] = group['eps']
    # dy[(dy < 0) & (dy > -group['eps'])] = -group['eps']
    # cg_param = inner(grad, y) / dy

    cg_param.add_(inner(y, y) * inner(grad, d_buf) / (dy ** 2 + eps), alpha=-group['lam'])
    # cg_param.add_(inner(y, y) * inner(grad, d_buf) / dy ** 2, alpha=-group['lam'])
    _eta = torch.min(inner(g_buf, g_buf), torch.full_like(cg_param, group['eps']))
    eta = -1 / (inner(d_buf, d_buf) * _eta)
    return torch.max(cg_param, eta)


_cg_param_dict = dict(
    HS=cg_param_hs,
    FR=cg_param_fr,
    PRP=cg_param_prp,
    DY=cg_param_dy,
    HZ=cg_param_hz,
)
