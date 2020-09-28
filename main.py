from typing import Any, Dict, Tuple, Union

from experiment.cifar10 import ExperimentCIFAR10
from experiment.imdb import ExperimentIMDb
from experiment.mnist import ExperimentMNIST
from experiment.stl10 import ExperimentSTL10
from optimizer.adam import Adam
from optimizer.conjugate.conjugate_momentum_adam import ConjugateMomentumAdam
from optimizer.conjugate.coba import CoBA

Optimizer = Union[Adam, CoBA, ConjugateMomentumAdam]
OptimizerDict = Dict[str, Tuple[Any, Dict[str, Any]]]


def prepare_optimizers(lr: float) -> OptimizerDict:
    return dict(
        CMAdam_HS=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=False, cg_type='HS')),
        CMAdam_FR=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=False, cg_type='FR')),
        CMAdam_PRP=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=False, cg_type='PRP')),
        CMAdam_DY=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=False, cg_type='DY')),
        CMAdam_HZ=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=False, cg_type='HZ')),

        CMAMSGrad_HS=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=True, cg_type='HS')),
        CMAMSGrad_FR=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=True, cg_type='FR')),
        CMAMSGrad_PRP=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=True, cg_type='PRP')),
        CMAMSGrad_DY=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=True, cg_type='DY')),
        CMAMSGrad_HZ=(ConjugateMomentumAdam, dict(lr=lr, amsgrad=True, cg_type='HZ')),

        CoBAdam_HS=(CoBA, dict(lr=lr, amsgrad=False, cg_type='HS')),
        CoBAdam_FR=(CoBA, dict(lr=lr, amsgrad=False, cg_type='FR')),
        CoBAdam_PRP=(CoBA, dict(lr=lr, amsgrad=False, cg_type='PRP')),
        CoBAdam_DY=(CoBA, dict(lr=lr, amsgrad=False, cg_type='DY')),
        CoBAdam_HZ=(CoBA, dict(lr=lr, amsgrad=False, cg_type='HZ')),

        CoBAMSGrad_HS=(CoBA, dict(lr=lr, amsgrad=True, cg_type='HS')),
        CoBAMSGrad_FR=(CoBA, dict(lr=lr, amsgrad=True, cg_type='FR')),
        CoBAMSGrad_PRP=(CoBA, dict(lr=lr, amsgrad=True, cg_type='PRP')),
        CoBAMSGrad_DY=(CoBA, dict(lr=lr, amsgrad=True, cg_type='DY')),
        CoBAMSGrad_HZ=(CoBA, dict(lr=lr, amsgrad=True, cg_type='HZ')),
    )


def prepare_optimizers_test(lr: float):
    return dict(
        Adam_Existing=(Adam, dict(lr=lr, amsgrad=False)),
        AMSGrad_Existing=(Adam, dict(lr=lr, amsgrad=True)),

        CoBAdamConst_HS=(CoBA, dict(lr=lr, amsgrad=False, cg_type='HS', a=1)),
        CoBAdamConst_FR=(CoBA, dict(lr=lr, amsgrad=False, cg_type='FR', a=1)),
        CoBAdamConst_PRP=(CoBA, dict(lr=lr, amsgrad=False, cg_type='PRP', a=1)),
        CoBAdamConst_DY=(CoBA, dict(lr=lr, amsgrad=False, cg_type='DY', a=1)),
        CoBAdamConst_HZ=(CoBA, dict(lr=lr, amsgrad=False, cg_type='HZ', a=1)),
    )


def imdb() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentIMDb(dataset_name='imdb', max_epoch=100, batch_size=32)
    e.execute(optimizers)


def mnist() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentMNIST(dataset_name='mnist', max_epoch=10, batch_size=32)
    e.execute(optimizers)


def cifar10() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentCIFAR10(dataset_name='cifar10', max_epoch=200, batch_size=128, model_name='ResNet44')
    e(optimizers)


def stl10() -> None:
    lr = 1e-3
    optimizers = prepare_optimizers(lr)
    e = ExperimentSTL10(dataset_name='stl10', max_epoch=200, batch_size=128, model_name='Inception3')
    e(optimizers)


if __name__ == '__main__':
    from sys import argv

    d = dict(
        imdb=imdb,
        mnist=mnist,
        cifar10=cifar10,
        stl10=stl10,
    )
    d[argv[1]]()
