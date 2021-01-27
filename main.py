from typing import *
# from warnings import simplefilter
# simplefilter('error')

from torch.optim.lr_scheduler import *
from torch.optim.sgd import SGD
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop

from experiment.avazu import ExperimentAvazu
from experiment.cifar10 import ExperimentCIFAR10
from experiment.coco import ExperimentCOCO
from experiment.imdb import ExperimentIMDb
from experiment.mnist import ExperimentMNIST
from experiment.stl10 import ExperimentSTL10
from experiment.svhn import ExperimentSVHN
from optimizer.adam import Adam
from optimizer.conjugate.conjugate_momentum_adam import ConjugateMomentumAdam
from optimizer.conjugate.coba import CoBA
from optimizer.conjugate.coba2 import CoBA2

Optimizer = Union[Adam, CoBA, ConjugateMomentumAdam]
OptimizerDict = Dict[str, Tuple[Any, Dict[str, Any]]]


def prepare_optimizers(lr: float, **kwargs) -> OptimizerDict:
    types = ('HS', 'FR', 'PRP', 'DY', 'HZ')
    kw_const = dict(a=1, m=1)
    # m_dict = dict(m2=1e-2, m3=1e-3, m4=1e-4)
    m_dict = dict(m0=1)
    # a_dict = dict(a4=1+1e-4, a5=1+1e-5, a6=1+1e-6, a7=1+1e-7)
    a_dict = dict(a8=1+1e-8)
    return dict(
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True, **kwargs)),
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False, **kwargs)),
        **{f'CoBAMSGrad_{t}_{sm}_{sa}': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, m=m, a=a, **kwargs))
           for t in types for sm, m in m_dict.items() for sa, a in a_dict.items()},
        # **{f'CoBAMSGrad2_{t}': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t)) for t in types},
        # **{f'CoBAMSGrad_{t}(const)': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
        # **{f'CoBAMSGrad2_{t}(const)': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
        Momentum_Existing=(SGD, dict(lr=lr, momentum=.9, **kwargs)),
        AdaGrad_Existing=(Adagrad, dict(lr=lr, **kwargs)),
        RMSProp_Existing=(RMSprop, dict(lr=lr, **kwargs)),
    )


def lr_lambda(epoch: int, max_epoch: int, lr: float, c: float = 1e-2):
    p = epoch / max_epoch
    if p < .1:
        return (10 * (1 - c) * p + c) * lr
    elif p < .5:
        return lr
    elif p < .75:
        return lr * 1e-1
    else:
        return lr * 1e-2


def avazu(max_epoch=30, lr=1e-2, batch_size=4096, num_workers=0, use_scheduler=False, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    scheduler = LambdaLR if use_scheduler else None
    kw_scheduler = dict(lr_lambda=lambda epoch: lr_lambda(epoch, max_epoch, lr))
    e = ExperimentAvazu(max_epoch=max_epoch, batch_size=batch_size, kw_loader=dict(num_workers=num_workers), 
                        scheduler=scheduler, kw_scheduler=kw_scheduler, **kwargs)
    e.execute(optimizers)


def imdb(lr=1e-3, max_epoch=100, batch_size=32, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentIMDb(max_epoch=max_epoch, batch_size=batch_size, **kwargs)
    e.execute(optimizers)


def mnist(lr=1e-3, max_epoch=100, batch_size=32, model_name='Perceptron2', use_scheduler=False, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    scheduler = ReduceLROnPlateau if use_scheduler else None
    e = ExperimentMNIST(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name, scheduler=scheduler,
                        **kwargs)
    e.execute(optimizers)


def cifar10(max_epoch=200, lr=1e-1, weight_decay=1e-4, batch_size=128, model_name='DenseNetBC24', num_workers=0,
            use_scheduler=False, **kwargs) -> None:
    scheduler = LambdaLR if use_scheduler else None
    kw_scheduler = dict(lr_lambda=lambda epoch: lr_lambda(epoch, max_epoch, lr))
    optimizers = prepare_optimizers(lr=lr, weight_decay=weight_decay)
    e = ExperimentCIFAR10(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name,
                          kw_loader=dict(num_workers=num_workers), scheduler=scheduler, kw_scheduler=kw_scheduler, 
                          **kwargs)
    e(optimizers)


def coco(max_epoch=100, lr=1e-3, batch_size=16, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentCOCO(max_epoch=max_epoch, batch_size=batch_size, **kwargs)
    e(optimizers)


def stl10(lr=1e-3, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentSTL10(model_name='Inception3', **kwargs)
    e(optimizers)


def svhn(lr=1e-3, max_epoch=50, batch_size=128, weight_decay=1e-4, model_name='DenseNetBC24', use_scheduler=False,
         **kwargs) -> None:
    scheduler = LambdaLR if use_scheduler else None
    kw_scheduler = dict(lr_lambda=lambda epoch: lr_lambda(epoch, max_epoch, lr))
    optimizers = prepare_optimizers(lr=lr, weight_decay=weight_decay)
    e = ExperimentSVHN(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name, scheduler=scheduler, 
                       kw_scheduler=kw_scheduler,**kwargs)
    e(optimizers)


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-e', '--experiment')
    p.add_argument('-m', '--model_name')
    p.add_argument('-d', '--data_dir', default='dataset/data')
    p.add_argument('-me', '--max_epoch', type=int)
    p.add_argument('-bs', '--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--device')
    p.add_argument('-nw', '--num_workers', type=int)
    p.add_argument('-us', '--use_scheduler', action='store_true')
    args = p.parse_args()

    experiment = args.experiment
    kw = {k: v for k, v in dict(**args.__dict__).items() if k != 'experiment' and v}

    d: Dict[str, Callable] = dict(
        Avazu=avazu,
        IMDb=imdb,
        CIFAR10=cifar10,
        MNIST=mnist,
        STL10=stl10,
        SVHN=svhn,
        COCO=coco,
    )
    d[experiment](**kw)
