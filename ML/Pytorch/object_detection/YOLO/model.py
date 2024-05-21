import torch
import torch.nn as nn
from torchsummary import summary

from data.darknet.darknet import Darknet


def _create_fcs(split_size, num_boxes, num_classes):
    S, B, C = split_size, num_boxes, num_classes

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024 * S * S, 496),
        # nn.Dropout(0.5),
        nn.LeakyReLU(0.1),
        nn.Linear(496, S * S * (C + B * 5)),
    )


class Yolov1(nn.Module):
    def __init__(self, **kwargs):
        super(Yolov1, self).__init__()
        self.backbone = Darknet("data/darknet/yolov1.cfg")
        self.backbone.load_state_dict(torch.load("data/darknet/yolov1-pytorch.pth"))
        self.fcs = _create_fcs(**kwargs)

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.fcs(x)
        return x


if __name__ == "__main__":
    model = Yolov1(split_size=7, num_boxes=3, num_classes=11)
    summary(model, (3, 448, 448))
