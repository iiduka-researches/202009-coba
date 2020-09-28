import torch


def f(x):
    return .5 * (x**2).sum()


def fprime(x):
    return x


def line_search(x, alpha_0, c1=1e-4, c2=.9):
    alpha = alpha_0
    while satisfies_armijo(alpha, ):
        alpha =
    return alpha


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1, keepdim=True)


def satisfies_armijo(closure, param, grad, alpha, d, c1: float) -> bool:
    if c1 <= 0 or c1 >= 1:
        raise ValueError(f'c1 should be 0 < c1 < 1, but c1= {c1}.')
    return closure(param + alpha * d) <= f(param) + c1 * alpha * inner(grad, param)


def satisfies_wolfe(alpha, c1: float, c2: float) -> bool:
    pass


if __name__ == '__main__':
    max_epoch = 100
    param_dim = 10
    lr = 1e-3

    torch.manual_seed(0)
    p = torch.rand(param_dim)

    for epoch in range(max_epoch):
        z = f(p)
        g = fprime(p)
        p.add_(-g, alpha=lr)
        if epoch % 10 == 0:
            print(f'objective: {z}')
            print('params:', p)
