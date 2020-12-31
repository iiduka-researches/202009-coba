from typing import Any, Dict, Tuple, Union
from warnings import simplefilter
simplefilter('error')

from torch.optim.sgd import SGD
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop

from experiment.avazu import ExperimentAvazu
from experiment.cifar10 import ExperimentCIFAR10
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
    kw_coefs = [dict(m=m, a=a) for m in (1e-2, 1e-3, 1e-4) for a in (1+1e-4, 1+1e-5, 1+1e-6, 1+1e-7)]
    return dict(
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False)),
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True)),
        **{f'CoBAMSGrad_{t}': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, **kw))
           for t in types for kw in kw_coefs},
        # **{f'CoBAMSGrad2_{t}': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t)) for t in types},
        # **{f'CoBAMSGrad_{t}(const)': (CoBA, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
        # **{f'CoBAMSGrad2_{t}(const)': (CoBA2, dict(lr=lr, amsgrad=True, cg_type=t, **kw_const)) for t in types},
    )


def avazu() -> None:
    optimizers = prepare_optimizers(lr=1e-4)
    e = ExperimentAvazu(max_epoch=10, batch_size=1024, num_workers=4, pin_memory=True)
    e.execute(optimizers)


def imdb() -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentIMDb(dataset_name='imdb', max_epoch=100, batch_size=32)
    e.execute(optimizers)


def mnist() -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentMNIST(max_epoch=10, batch_size=32)
    e.execute(optimizers)


def cifar10(model='DenseNetBC24') -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentCIFAR10(max_epoch=300, batch_size=256, model_name=model)
    e(optimizers)


def stl10() -> None:
    optimizers = prepare_optimizers(lr=1e-3)
    e = ExperimentSTL10(dataset_name='stl10', max_epoch=200, batch_size=128, model_name='Inception3')
    e(optimizers)


if __name__ == '__main__':
    from sys import argv

    d = dict(
        avazu=avazu,
        imdb=imdb,
        cifar10=cifar10,
        # stl10=stl10,
    )
    experiment = argv[1]
    kw: Dict[str, Any] = dict()
    if len(argv) > 2:
        kw['model'] = argv[2]
    d[experiment](**kw)
