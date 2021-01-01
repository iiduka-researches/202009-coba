import os
from typing import List, Tuple, Optional

import torch.tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torchvision.datasets import CocoDetection
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn

import sys
sys.path.append('.')
sys.path.append('..')
from optimizer.base_optimizer import Optimizer
from experiment.base import BaseExperiment, ResultDict


def collate_fn(batch):
    return tuple(zip(*batch))


def arrange_box(b: List[float]) -> List[float]:
    b[2] += b[0]
    b[3] += b[1]
    return b[:]


def target_transform(target):
    """
    {'segmentation': [[587.72, 58.39, 578.34, 57.65, 572.41, 55.92, 573.64, 51.47, 582.29, 47.28, 586.24, 48.26]],
     'area': 120.21155000000036, 'iscrowd': 0, 'image_id': 510078, 'bbox': [572.41, 47.28, 15.31, 11.11],
     'category_id': 3, 'id': 2046811}
    """
    return [dict(boxes=torch.tensor(arrange_box(t['bbox']), dtype=torch.float).reshape(1, -1),
                 labels=torch.tensor(t['category_id'], dtype=torch.int).reshape(1, -1)) for t in target]


class ExperimentCOCO(BaseExperiment):
    def __init__(self, **kwargs) -> None:
        super(ExperimentCOCO, self).__init__(dataset_name='coco',
                                             kw_dataset=dict(transform=ToTensor(), target_transform=target_transform),
                                             kw_loader=dict(collate_fn=collate_fn), **kwargs)

    def prepare_data(self, train: bool, **kwargs) -> Dataset:
        root = os.path.join(self.data_dir, 'coco')
        os.makedirs(root, exist_ok=True)

        # check and download
        if not os.path.isdir(os.path.join(root, 'train2017')):
            download_and_extract_archive('http://images.cocodataset.org/zips/train2017.zip', root)
        if not os.path.isdir(os.path.join(root, 'val2017')):
            download_and_extract_archive('http://images.cocodataset.org/zips/val2017.zip', root)
        if not os.path.isdir(os.path.join(root, 'annotations')):
            download_and_extract_archive('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', root)

        if train:
            return CocoDetection(root=os.path.join(root, 'train2017'),
                                 annFile=os.path.join(root, 'annotations/instances_train2017.json'), **kwargs)
        else:
            return CocoDetection(root=os.path.join(root, 'val2017'),
                                 annFile=os.path.join(root, 'annotations/instances_val2017.json'), **kwargs)

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name:
            return fasterrcnn_resnet50_fpn(pretrained=False, progress=False, num_classes=91,
                                           pretrained_backbone=False, trainable_backbone_layers=5, **kwargs)
        else:
            raise ValueError(f'model_name: {model_name}.')

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader) -> Tuple[Module, ResultDict]:
        for images, targets in train_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for target in targets for t in target]
            optimizer.zero_grad()
            outputs = net(images, targets)
            break

        return net, outputs

    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        pass
