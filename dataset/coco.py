from math import ceil
import os.path
import pickle
from typing import List

import torch
from torch import tensor
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision.datasets import CocoDetection
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import Compose, Pad, Resize, ToTensor
from tqdm import tqdm


def _load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def _dump_pickle(data, path):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)


def arrange_box(b: List[float], pad) -> List[float]:
    b[2] += b[0]
    b[3] += b[1]
    return b[:]


def _pad_to_square(image):
    # padding both side
    _, h, w = image.shape
    dim_diff = abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    padding = (0, pad1, 0, pad2) if h <= w else (pad1, 0, pad2, 0)
    # Add padding
    image = F.pad(image, padding)

    return image, padding


def pad_to_square(image: torch.Tensor):
    # padding below or right (only one side)
    _, h, w = image.shape
    dim_diff = abs(h - w)
    padding = (0, 0, 0, dim_diff) if h <= w else (0, 0, dim_diff, 0)
    # Add padding
    image = F.pad(image, padding)
    return image, padding


def transforms(image, target, image_size: int):
    image = ToTensor()(image)
    image, pad = pad_to_square(image)
    image = Resize((image_size, image_size))(image)
    scale = image_size / image.size(1)

    _target = dict()
    if target:
        _target['boxes'] = tensor([arrange_box(t['bbox'], pad) for t in target], dtype=torch.float) * scale
        _target['labels'] = tensor([t['category_id'] for t in target], dtype=torch.int64)
        _target['image_id'] = tensor([target[0]['image_id']])
    # _target['masks'] = []

    return image, dict(**_target)


class COCODataset(Dataset):
    def __init__(self, root: str, train=True, image_size=416, **kwargs) -> None:
        self.image_size = image_size

        # check and download
        train_dir = os.path.join(root, 'train2017')
        test_dir = os.path.join(root, 'val2017')
        annotation_dir = os.path.join(root, 'annotations')
        if not os.path.isdir(train_dir):
            download_and_extract_archive('http://images.cocodataset.org/zips/train2017.zip', root)
        if not os.path.isdir(test_dir):
            download_and_extract_archive('http://images.cocodataset.org/zips/val2017.zip', root)
        if not os.path.isdir(annotation_dir):
            download_and_extract_archive('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                         root)
        if train:
            _root = train_dir
            ann_file = os.path.join(annotation_dir, 'instances_train2017.json')
        else:
            _root = test_dir
            ann_file = os.path.join(annotation_dir, 'instances_val2017.json')
        self.coco_detection = CocoDetection(root=_root, annFile=ann_file, transforms=self.transforms, **kwargs)
        self.coco = self.coco_detection.coco
        self.ids = self.coco_detection.ids

    def __getitem__(self, index):
        return self.coco_detection.__getitem__(index)

    def __len__(self) -> int:
        return len(self.ids)

    def transforms(self, image, target):
        return transforms(image, target, image_size=self.image_size)
