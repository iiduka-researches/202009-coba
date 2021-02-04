from typing import Sequence, Tuple, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import sigmoid
from torch.nn import BCEWithLogitsLoss, Embedding, Linear, Module
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchfm.model.dfm import DeepFactorizationMachineModel
from tqdm import tqdm

from .base import BaseExperiment, ResultDict
from dataset.avazu import AvazuDataset


class ExperimentAvazu(BaseExperiment):
    def __init__(self, embedding_dim=20, model_name='LogisticRegression', weight_decay=1e-6, **kwargs) -> None:
        super(ExperimentAvazu, self).__init__(dataset_name='Avazu', model_name=model_name,
                                              kw_optimizer=dict(weight_decay=weight_decay), **kwargs)
        self.embedding_dim = embedding_dim

    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        return AvazuDataset(train=train, **kwargs)

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        sizes = tuple(AvazuDataset.size_dict.values())
        embedding_dims = [self.embedding_dim for _ in sizes]
        if model_name == 'LogisticRegression':
            return LinearRegression(sizes=sizes, embedding_dims=embedding_dims, device=self.device, **kwargs)
        elif model_name:
            return _MODEL_DICT[model_name](sizes, embed_dim=self.embedding_dim, mlp_dims=(16, 16), dropout=0.2,
                                           **kwargs)
        else:
            raise ValueError(f'model_name: {model_name} is invalid.')

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: data.DataLoader,
                    **kwargs) -> Tuple[Module, ResultDict]:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        label_list, prob_list = [], []
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(self.device, dtype=torch.long)
            labels = labels.to(self.device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = torch.reshape(net(inputs), (-1, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=None)

            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.where(outputs <= .5, 0, 1)
            correct += (predicted == labels).sum().item()

            label_list.append(labels.detach().cpu().numpy().flatten())
            prob_list.append(sigmoid(outputs).detach().cpu().numpy().flatten())
            i += 1

        y_true = np.hstack(label_list)
        y_pred = np.hstack(prob_list)
        auc = roc_auc_score(y_true, y_pred)
        from utils.line.notify import notify
        notify(f'Loss:\t{running_loss / i}\nAcc:\t{correct / total}\nAUC:\t{auc}.')
        return net, dict(train_loss=running_loss / i, train_accuracy=correct / total, train_auc=auc)

    def epoch_validate(self, net: Module, test_loader: data.DataLoader, **kwargs) -> ResultDict:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        label_list, prob_list = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device, dtype=torch.long)
                labels = labels.to(self.device, dtype=torch.float)
                outputs = torch.reshape(net(inputs), (-1, 1))
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = torch.where(outputs <= .5, 0, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                label_list.append(labels.detach().cpu().numpy().flatten())
                prob_list.append(sigmoid(outputs).detach().cpu().numpy().flatten())

                i += 1

        y_true = np.hstack(label_list)
        y_pred = np.hstack(prob_list)
        auc = roc_auc_score(y_true, y_pred)

        return dict(test_loss=running_loss / i, test_accuracy=correct / total, test_auc=auc)


class LinearRegression(Module):
    def __init__(self, sizes: Sequence[int], embedding_dims: Sequence[int], device, out_dim=1) -> None:
        super(LinearRegression, self).__init__()
        self.embeddings = [Embedding(num_embeddings, embedding_dim).to(device=device)
                           for num_embeddings, embedding_dim in zip(sizes, embedding_dims)]
        self.linear = Linear(sum(embedding_dims), out_dim)

    def forward(self, x):
        m = torch.cat([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], 1)
        return self.linear(m)


_MODEL_DICT = dict(
    LogisticRegression=LinearRegression,
    DeepFM=DeepFactorizationMachineModel,
)
