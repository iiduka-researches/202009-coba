import torch


def get_cg_param_fn(cg_type: str):
    return _cg_param_dict[cg_type]


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def cg_param_hs(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    y = g - past_g
    dy = inner(past_cg, y)
    return inner(g, y) / (dy + torch.where(dy >= 0, eps, -eps))


def cg_param_fr(g: torch.Tensor, past_g: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    past_g_norm_sq = inner(past_g, past_g)
    return inner(g, g) / (past_g_norm_sq + eps)


def cg_param_prp(g: torch.Tensor, past_g: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    y = g - past_g
    past_g_norm_sq = inner(past_g, past_g)
    return inner(g, y) / (past_g_norm_sq + eps)


def cg_param_dy(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    y = g - past_g
    dy = inner(past_cg, y)
    return inner(g, g) / (dy + torch.where(dy >= 0, eps, -eps))


def cg_param_cd(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    r"""
    proposed by Fletcher, CD stands for “Conjugate Descent”
    """
    dg = inner(past_cg, past_g)
    return - inner(g, g) / (dg + torch.where(dg >= 0, eps, -eps))


def cg_param_ls(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    r"""
    proposed by Liu and Storey
    """
    y = g - past_g
    dg = inner(past_cg, past_g)
    return - inner(g, y) / (dg + torch.where(dg >= 0, eps, -eps))


def cg_param_pseudo_deterministic_hs(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps,
                                     **kwargs) -> torch.Tensor:
    y = g - past_g
    dy = inner(past_cg, y)
    return inner(past_g, y) / (dy + torch.where(dy >= 0, eps, -eps))


def cg_param_pseudo_deterministic_fr(g: torch.Tensor, past_g: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    past_g_norm_sq = inner(past_g, past_g)
    return inner(g, past_g) / (past_g_norm_sq + eps)


def cg_param_pseudo_deterministic_prp(g: torch.Tensor, past_g: torch.Tensor, eps, **kwargs) -> torch.Tensor:
    y = g - past_g
    past_g_norm_sq = inner(past_g, past_g)
    return inner(past_g, y) / (past_g_norm_sq + eps)


def cg_param_pseudo_deterministic_dy(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps,
                                     **kwargs) -> torch.Tensor:
    y = g - past_g
    dy = inner(past_cg, y)
    return inner(g, past_g) / (dy + torch.where(dy >= 0, eps, -eps))


def cg_param_pseudo_deterministic_cd(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps,
                                     **kwargs) -> torch.Tensor:
    r"""
    proposed by Fletcher, CD stands for “Conjugate Descent”
    """
    dg = inner(past_cg, past_g)
    return - inner(g, past_g) / (dg + torch.where(dg >= 0, eps, -eps))


def cg_param_pseudo_deterministic_ls(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps,
                                     **kwargs) -> torch.Tensor:
    r"""
    proposed by Liu and Storey
    """
    y = g - past_g
    dg = inner(past_cg, past_g)
    return - inner(past_g, y) / (dg + torch.where(dg >= 0, eps, -eps))


def cg_param_hz(g: torch.Tensor, past_g: torch.Tensor, past_cg: torch.Tensor, eps, lam, **kwargs) -> torch.Tensor:
    y = g - past_g
    dy = inner(past_cg, y)
    cg_param = inner(g, y) / (dy + torch.where(dy >= 0, eps, -eps))

    cg_param.add_(inner(y, y) * inner(g, past_cg) / (dy ** 2 + eps),
                  alpha=-lam)
    # _eta = torch.min(inner(g, g), torch.full_like(cg_param, eps))
    # eta = -1 / (inner(past_cg, past_cg) * _eta)
    # return torch.max(cg_param, eta)
    return cg_param


_cg_param_dict = dict(
    HS=cg_param_hs,
    FR=cg_param_fr,
    PRP=cg_param_prp,
    DY=cg_param_dy,
    CD=cg_param_cd,
    LS=cg_param_ls,
    PdHS=cg_param_pseudo_deterministic_hs,
    PdFR=cg_param_pseudo_deterministic_fr,
    PdPRP=cg_param_pseudo_deterministic_prp,
    PdDY=cg_param_pseudo_deterministic_dy,
    PdCD=cg_param_pseudo_deterministic_cd,
    PdLS=cg_param_pseudo_deterministic_ls,
    # HZ=cg_param_hz,
)
