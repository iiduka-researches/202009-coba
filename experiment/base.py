from abc import ABCMeta, abstractmethod
from datetime import datetime
import os
import random
from time import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pandas import DataFrame
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from optimizer.base_optimizer import Optimizer
from optimizer.conjugate.coba import CoBA
from optimizer.conjugate.coba2 import CoBA2
from utils.line.notify import notify, notify_error

ParamDict = Dict[str, Any]
OptimDict = Dict[str, Tuple[Any, ParamDict]]
ResultDict = Dict[str, float]
Result = Dict[str, Sequence[float]]

SEP = '_'


class BaseExperiment(metaclass=ABCMeta):
    def __init__(self, batch_size: int, max_epoch: int, dataset_name: str, kw_dataset=None, kw_loader=None,
                 model_name='model', kw_model=None, kw_optimizer=None, data_dir='./dataset/data/') -> None:
        r"""Base class for all experiments.

        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.data_dir = data_dir
        self.device = select_device()
        _kw_dataset = kw_dataset if kw_dataset else dict()
        self.train_data = self.prepare_data(train=True, **_kw_dataset)
        self.test_data = self.prepare_data(train=False, **_kw_dataset)
        self.kw_loader = kw_loader if kw_loader else dict()
        self.model_name = model_name
        self.kw_model = kw_model if kw_model else dict()
        self.kw_optimizer = kw_optimizer if kw_optimizer else dict()

    def __call__(self, *args, **kwargs) -> None:
        self.execute(*args, **kwargs)

    @abstractmethod
    def prepare_data(self, train: bool, **kwargs) -> Dataset:
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        raise NotImplementedError

    @abstractmethod
    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        raise NotImplementedError

    @abstractmethod
    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        raise NotImplementedError

    def train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader,
              test_loader: DataLoader) -> Tuple[Module, Result]:
        results = []
        for epoch in tqdm(range(self.max_epoch)):
            start = time()
            net, train_result = self.epoch_train(net, optimizer=optimizer, train_loader=train_loader)
            validate_result = self.epoch_validate(net, test_loader=test_loader)
            result = arrange_result_as_dict(t=time() - start, train=train_result, validate=validate_result)
            results.append(result)
            if epoch % 10 == 0:
                notify(str(result))
        return net, concat_dicts(results)

    @notify_error
    def execute(self, optimizers: OptimDict, result_dir='./result', seed=0) -> None:
        model_dir = os.path.join(result_dir, self.dataset_name, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        train_loader = DataLoader(self.train_data,  batch_size=self.batch_size, shuffle=True,
                                  worker_init_fn=worker_init_fn, **self.kw_loader)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                 worker_init_fn=worker_init_fn, **self.kw_loader)
        period = len(train_loader)
        for name, (optimizer_cls, kw_optimizer) in optimizers.items():
            path = os.path.join(model_dir, result_format(name))

            if exist_result(name, model_dir):
                notify(f'{name} already exists.')
                continue
            else:
                notify(f'{name}')

            fix_seed(seed)
            if 'CoBA' in name:
                kw_optimizer['period'] = period
            net = self.prepare_model(self.model_name, **self.kw_model)
            net.to(self.device)
            optimizer = optimizer_cls(net.parameters(), **kw_optimizer, **self.kw_optimizer)
            _, result = self.train(net=net, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader)
            result_to_csv(result, name=name, kw_optimizer=optimizer.__dict__.get('defaults', kw_optimizer),
                          path=path)

            # Expect error between Stochastic CG and Deterministic CG
            if type(optimizer) in (CoBA, CoBA2):
                s = '\n'.join([str(e) for e in optimizer.scg_expect_errors])
                with open(os.path.join(model_dir, f'scg_expect_errors_{name}.csv'), 'w') as f:
                    f.write(s)


def select_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Using {device} ...')
    return device


def arrange_result_as_dict(t: float, train: Dict[str, float], validate: Dict[str, float]) -> Dict[str, float]:
    train = {k if 'train' in k else f'train_{k}': v for k, v in train.items()}
    validate = {k if 'test' in k else f'test_{k}': v for k, v in validate.items()}
    return dict(time=t, **train, **validate)


def concat_dicts(results: Sequence[ResultDict]) -> Result:
    keys = results[0].keys()
    return {k: [r[k] for r in results] for k in keys}


def fix_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    random.seed(worker_id)


def result_format(name: str, sep=SEP, extension='csv') -> str:
    ts = datetime.now().strftime('%y%m%d%H%M%S')
    return f'{name}{sep}{ts}.{extension}'


def exist_result(name: str, result_dir: str, sep=SEP) -> bool:
    for p in os.listdir(result_dir):
        if SEP.join(os.path.basename(p).split(sep)[:-1]) == name:
            return True
    return False


def result_to_csv(r: Result, name: str, kw_optimizer: ParamDict, path: str) -> None:
    df = DataFrame(r)
    df['optimizer'] = name
    df['optimizer_parameters'] = str(kw_optimizer)
    df['epoch'] = np.arange(1, df.shape[0] + 1)
    df.set_index(['optimizer', 'optimizer_parameters', 'epoch'], drop=True, inplace=True)
    df.to_csv(path, encoding='utf-8')
