import torch
import torch.nn as nn

from model.yolo.layer import Bottleneck, Concat, Conv, SPP


ANCHORS = (
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326],  # P5/32
)

# darknet53 backbone
BACKBONE = (
    # [from, number, module, args]
    [-1, 1, Conv, [32, 3, 1]],  # 0
    [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
    [-1, 1, Bottleneck, [64]],
    [-1, 1, Conv, [128, 3, 2]],  # 3-P2/4
    [-1, 2, Bottleneck, [128]],
    [-1, 1, Conv, [256, 3, 2]],  # 5-P3/8
    [-1, 8, Bottleneck, [256]],
    [-1, 1, Conv, [512, 3, 2]],  # 7-P4/16
    [-1, 8, Bottleneck, [512]],
    [-1, 1, Conv, [1024, 3, 2]],  # 9-P5/32
    [-1, 4, Bottleneck, [1024]],  # 10
)


class Detect(nn.Module):
    """
    Attributes
    ----------
    num_classes: int
        number of classes
    num_outputs: int
        number of outputs per anchor
    num_layers: int
        number of detection layers
    num_anchors: int
        number of anchors
    """
    stride = None  # strides computed during build

    def __init__(self, num_classes: int, anchors, in_channels=()) -> None:
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.num_layers = len(anchors)
        self.num_anchors = len(anchors[0])
        self.grid = [torch.zeros(1)] * self.num_layers

        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.num_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)

        self.modules = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in in_channels)


class YOLOv3SPP(nn.Module):
    def __init__(self, num_classes=80, in_channels=3, anchors=ANCHORS, backbone=BACKBONE) -> None:
        super(YOLOv3SPP, self).__init__()

        # Define model
        self.num_classes = num_classes
        self.anchors = anchors
        self.backbone = backbone

        # YOLOv3-SPP head
        self.head = [
            [-1, 1, Bottleneck, [1024, False]],
            [-1, 1, SPP, [512, [5, 9, 13]]],
            [-1, 1, Conv, [1024, 3, 1]],
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, Conv, [1024, 3, 1]],  # 15 (P5/32-large)

            [-2, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 8], 1, Concat, [1]],  # cat backbone P4
            [-1, 1, Bottleneck, [512, False]],
            [-1, 1, Bottleneck, [512, False]],
            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, Conv, [512, 3, 1]],  # 22 (P4/16-medium)

            [-2, 1, Conv, [128, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, Concat, [1]],  # cat backbone P3
            [-1, 1, Bottleneck, [256, False]],
            [-1, 2, Bottleneck, [256, False]],  # 27 (P3/8-small)

            [[27, 22, 15], 1, Detect, [num_classes, anchors]],  # Detect(P3, P4, P5)
        ]

    def forward(self):
        pass
