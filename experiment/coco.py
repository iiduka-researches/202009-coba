import json
import os
from typing import Tuple, Optional

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import torch.tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn
from tqdm import tqdm

from dataset.coco import COCODataset
from experiment.base import BaseExperiment, ResultDict, Optimizer

ANN_FILE = 'dataset/data/coco2017/annotations/instances_val2017.json'
STATS_LABELS = (
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
    'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
)


def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    images = torch.stack(images)
    return images, targets


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def arrange(target, output):
    image_id = target["image_id"].item()
    boxes = convert_to_xywh(output['boxes']).tolist()
    scores = output['scores'].tolist()
    labels = output['labels'].tolist()
    return [dict(image_id=image_id, category_id=label, bbox=box, score=score)
            for label, box, score in zip(labels, boxes, scores)]


class ExperimentCOCO(BaseExperiment):
    def __init__(self, **kwargs) -> None:
        self.coco = None
        super(ExperimentCOCO, self).__init__(dataset_name='coco2017', **kwargs)
        self.kw_loader.setdefault('collate_fn', collate_fn)

    def prepare_data(self, train: bool, **kwargs) -> Dataset:
        dataset = COCODataset(root=self.data_dir, train=train, **kwargs)
        if self.coco is None:
            self.coco = COCO(os.path.join(self.data_dir, 'annotations/instances_val2017.json'))
        return dataset

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name:
            return fasterrcnn_resnet50_fpn(pretrained=False, progress=False, num_classes=91,
                                           pretrained_backbone=False, trainable_backbone_layers=5, **kwargs)
        else:
            raise ValueError(f'model_name: {model_name}.')

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: DataLoader, **kwargs) -> Tuple[Module, ResultDict]:
        """
        outputs = {
            'loss_classifier': tensor(5.7116, grad_fn=<NllLossBackward>),
            'loss_box_reg': tensor(0.0040, grad_fn=<DivBackward0>),
            'loss_objectness': tensor(0.6018, grad_fn=<BinaryCrossEntropyWithLogitsBackward>),
            'loss_rpn_box_reg': tensor(0.5090, grad_fn=<DivBackward0>)
        }
        """
        net.train()
        i = 0
        running_loss_dict: ResultDict = dict()
        for images, targets in tqdm(train_loader, total=len(train_loader)):
            images = list(image.to(self.device) for image, target in zip(images, targets) if target)
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets if target]
            loss_dict = net(images, targets)
            loss = torch.cat([l.reshape(1) for _, l in loss_dict.items()]).sum()
            running_loss_dict = {k: running_loss_dict.get(k, .0) + v.item() for k, v in loss_dict.items()}

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            i += 1

        return net, {f'train_{k}': v / i for k, v in running_loss_dict.items()}

    @torch.no_grad()
    def epoch_validate(self, net: Module, test_loader: DataLoader, **kwargs) -> ResultDict:
        net.eval()
        res = []
        for images, targets in test_loader:
            images = list(image.to(self.device) for image, target in zip(images, targets) if target)
            targets = [{k: v.to(self.device) for k, v in target.items()} for target in targets if target]

            outputs = net(images)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            res.extend([d for target, output in zip(targets, outputs) for d in arrange(target, output)])

        """
        coco_dt = self.coco.loadRes(res)
        coco_eval = COCOeval(self.coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        stats = coco_eval.stats
        return dict(zip(STATS_LABELS, stats))
        """
        return dict(test_result=json.dumps(res))
