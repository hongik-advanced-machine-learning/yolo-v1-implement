"""
Implementation of Yolo Loss Function from the original yolo paper

"""

import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    # def forward(self, predictions, target):
    #     # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
    #     predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
    #
    #     # confidence score가 더 높은 bbox만 학습에 사용
    #     # confidence score = Pr(Object) * IoU(truthpred)
    #     iou_b1 = intersection_over_union(predictions[..., self.C + 1:self.C + 5], target[..., self.C + 1:self.C + 5])
    #     iou_b2 = intersection_over_union(predictions[..., self.C + 6:self.C + 10], target[..., self.C + 1:self.C + 5])
    #     # shape=(2, BATCH_SIZE, S, S, 1)
    #     ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
    #
    #     # bestbox: iou_maxes가 위치한 인덱스([0, B)), shape=(BATCH_SIZE, S, S, 1)
    #     _, bestbox = torch.max(ious, dim=0)
    #     # grid cell 하나에 물체 하나만 매핑되어있기 때문에 self.C 위치의 값(0 or 1)이 물체의 존재 여부
    #     # shape=(BATCH_SIZE, S, S, 1)
    #     is_object_exist = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i
    #
    #     localization_loss = self._localization(predictions, target, is_object_exist, bestbox)
    #
    #     # ==================== #
    #     #   FOR OBJECT LOSS    #
    #     # ==================== #
    #
    #     # pred_box is the confidence score for the bbox with highest IoU
    #     pred_box = (
    #             bestbox * predictions[..., self.C + 5:self.C + 6] + (1 - bestbox) * predictions[..., self.C:self.C + 1]
    #     )
    #
    #     object_loss = self.mse(
    #         torch.flatten(is_object_exist * pred_box),
    #         torch.flatten(is_object_exist * target[..., self.C:self.C + 1]),
    #     )
    #
    #     # ======================= #
    #     #   FOR NO OBJECT LOSS    #
    #     # ======================= #
    #
    #     # max_no_obj = torch.max(predictions[..., self.C:self.C+1], predictions[..., self.C+5:self.C+6])
    #     # no_object_loss = self.mse(
    #     #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
    #     #    torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
    #     # )
    #
    #     no_object_loss = self.mse(
    #         torch.flatten((1 - is_object_exist) * predictions[..., self.C:self.C + 1], start_dim=1),
    #         torch.flatten((1 - is_object_exist) * target[..., self.C:self.C + 1], start_dim=1),
    #     )
    #
    #     no_object_loss += self.mse(
    #         torch.flatten((1 - is_object_exist) * predictions[..., self.C + 5:self.C + 6], start_dim=1),
    #         torch.flatten((1 - is_object_exist) * target[..., self.C:self.C + 1], start_dim=1)
    #     )
    #
    #     # ================== #
    #     #   FOR CLASS LOSS   #
    #     # ================== #
    #
    #     class_loss = self.mse(
    #         torch.flatten(is_object_exist * predictions[..., :self.C], end_dim=-2, ),
    #         torch.flatten(is_object_exist * target[..., :self.C], end_dim=-2, ),
    #     )
    #
    #     loss = (
    #             self.lambda_coord * localization_loss  # first two rows in paper
    #             + object_loss  # third row in paper
    #             + self.lambda_noobj * no_object_loss  # forth row
    #             + class_loss  # fifth row
    #     )
    #
    #     return loss

    def _localization(self,
                      predictions: torch.Tensor,
                      target: torch.Tensor,
                      is_object_exist: torch.Tensor,
                      bestbox: torch.Tensor):
        mask = (bestbox * 5 + torch.arange(1, 5, device=bestbox.device)).view(-1, self.S, self.S, 4)
        box_predictions = is_object_exist * torch.gather(predictions, -1, mask + self.C)

        box_targets = is_object_exist * target[..., self.C + 1:self.C + 5]

        # -0.0이 있어 nan이 나오는 경우가 있음
        box_predictions[..., 2:4] = torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(box_predictions, box_targets)
        return box_loss

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., self.C].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., self.C+6:self.C+10]
                + (1 - bestbox) * predictions[..., self.C+1:self.C+5]
            )
        )

        box_targets = exists_box * target[..., self.C+1:self.C+5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., self.C+5:self.C+6] + (1 - bestbox) * predictions[..., self.C:self.C+1]
        )

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        #max_no_obj = torch.max(predictions[..., self.C:self.C+1], predictions[..., self.C+5:self.C+6])
        #no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
        #)

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2,),
            torch.flatten(exists_box * target[..., :self.C], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss
