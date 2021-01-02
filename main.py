from typing import Any, Dict, Tuple, Union
from warnings import simplefilter
simplefilter('error')

from torch.optim.sgd import SGD
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop

from experiment.avazu import ExperimentAvazu
from experiment.cifar10 import ExperimentCIFAR10
from experiment.coco import ExperimentCOCO
from experiment.imdb import ExperimentIMDb
from experiment.mnist import ExperimentMNIST
from experiment.stl10 import ExperimentSTL10
from optimizer.adam import Adam
from optimizer.conjugate.conjugate_momentum_adam import ConjugateMomentumAdam
from optimizer.conjugate.coba import CoBA
from optimizer.conjugate.coba2 import CoBA2

Optimizer = Union[Adam, CoBA, ConjugateMomentumAdam]
OptimizerDict = Dict[str, Tuple[Any, Dict[str, Any]]]


def prepare_optimizers(lr: float) -> OptimizerDict:
    types = ('HS', 'FR', 'PRP', 'DY', 'HZ')
    kw_const = dict(a=1, m=1)
    # m_dict = dict(m2=1e-2, m3=1e-3, m4=1e-4)
    m_dict = dict(m3=1e-3)
    # a_dict = dict(a4=1+1e-4, a5=1+1e-5, a6=1+1e-6, a7=1+1e-7)
    a_dict = dict(a6=1+1e-6)
    return dict(
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False)),
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True)),
        Momentum_Existing=(SGD, dict(lr=lr, momentum=.9)),
        AdaGrad_Existing=(Adagrad, dict(lr=lr)),
        RMSProp_Existing=(RMSprop, dict(lr=lr)),
        **{f'CoBAMSGrad_{t}_{sm}_{sa}': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, m=m, a=a))
           for t in types for sm, m in m_dict.items() for sa, a in a_dict.items()},
        # **{f'CoBAMSGrad2_{t}': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t)) for t in types},
        # **{f'CoBAMSGrad_{t}(const)': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
        # **{f'CoBAMSGrad2_{t}(const)': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
    )


def avazu(max_epoch=10, lr=1e-4, batch_size=1024, num_workers=0, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentAvazu(max_epoch=max_epoch, batch_size=batch_size, kw_loader=dict(num_workers=num_workers), **kwargs)
    e.execute(optimizers)


def imdb(**kwargs) -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentIMDb(dataset_name='imdb', max_epoch=100, batch_size=32, **kwargs)
    e.execute(optimizers)


def mnist(**kwargs) -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentMNIST(max_epoch=10, batch_size=32, **kwargs)
    e.execute(optimizers)


def cifar10(max_epoch=200, lr=1e-3, batch_size=128, num_workers=0, model_name='DenseNetBC24', **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentCIFAR10(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name,
                          kw_loader=dict(num_workers=num_workers), **kwargs)
    e(optimizers)


def coco(max_epoch=100, lr=1e-3, batch_size=16, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentCOCO(max_epoch=max_epoch, batch_size=batch_size, **kwargs)
    e(optimizers)


def stl10(lr=1e-3) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentSTL10(dataset_name='stl10', model_name='Inception3')
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
    args = p.parse_args()

    experiment = args.experiment
    kw = {k: v for k, v in dict(**args.__dict__).items() if k != 'experiment' and v}
    d = dict(
        Avazu=avazu,
        IMDb=imdb,
        CIFAR10=cifar10,
        # STL10=stl10,
        COCO=coco,
    )

    d[experiment](**kw)
