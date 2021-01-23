from torch.utils import data
from torchvision.datasets import SVHN
from torchvision.transforms import ToTensor

from experiment.cifar10 import ExperimentCIFAR10


class ExperimentSVHN(ExperimentCIFAR10):
    def __init__(self, **kwargs) -> None:
        super(ExperimentSVHN, self).__init__(dataset_name='SVHN', **kwargs)

    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        if train:
            split = 'train'
        else:
            split = 'test'
        return SVHN(root=self.data_dir, split=split, download=True, transform=ToTensor(), **kwargs)