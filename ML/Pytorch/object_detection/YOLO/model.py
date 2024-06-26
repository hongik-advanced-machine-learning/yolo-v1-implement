import torch
import torch.nn as nn
from torchsummary import summary

from data.darknet.darknet import DarkNet


class YOLOv1(nn.Module):
    """
    input image size : 448*448
    output size : self.S*self.S * 30
        self.S means grid
        30 means self.B Bbox information(x,y,w,h,confidence) + class probability (self.C)
    """

    def __init__(self, **kwargs):
        super(YOLOv1, self).__init__()

        darknet = DarkNet("data/darknet/cfg/yolov1-tiny.cfg")
        darknet.load_weights("data/darknet/weight/tiny-yolov1.weights")

        self.S = kwargs["split_size"]
        self.B = kwargs["num_boxes"]
        self.C = kwargs["num_classes"]

        # Make layers
        self.features = darknet.features
        self.fc = self._make_fc_layers()

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, 5 * self.B + self.C)
        return x

    def _make_fc_layers(self):  ## 2 Fully connected
        fc = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=self.S * self.S * 256, out_features=4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),  # defalut inplace = False
            nn.Linear(in_features=4096, out_features=(self.S * self.S * (5 * self.B + self.C))),
        )
        return fc


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def test():
    yolo = YOLOv1(split_size=7, num_boxes=2, num_classes=20)

    print(yolo)
    summary(yolo, (3, 448, 448))


if __name__ == '__main__':
    test()
