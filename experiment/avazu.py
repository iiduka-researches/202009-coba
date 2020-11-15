from typing import Tuple, Optional

from torch.nn import Module, Linear, Embedding
from torch.utils.data import DataLoader

from optimizer.base_optimizer import Optimizer
from .base_experiment import BaseExperiment, ResultDict


class ExperimentAvazu(BaseExperiment):
    def prepare_data_loader(self, batch_size: int, data_dir: str) -> Tuple[DataLoader, DataLoader, dict]:
        pass

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name:
            return _MODEL_DICT[model_name](**kwargs)
        else:
            raise ValueError(f'model_name: {model_name} is invalid.')

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        pass

    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        pass


class LinearRegression(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int) -> None:
        super(LinearRegression, self).__init__()
        self.emb = Embedding(num_embeddings, embedding_dim)
        self.linear = Linear(embedding_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


_MODEL_DICT = dict(
    logistic_regression=LinearRegression
)
