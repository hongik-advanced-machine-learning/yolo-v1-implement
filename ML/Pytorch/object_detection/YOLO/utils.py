from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms(bboxes: list[list[int | float]],
        iou_threshold: float,
        threshold: float,
        box_format="corners") -> list[list[int | float]]:
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1])

    dict = defaultdict(list)
    for box in bboxes:
        dict[box[0]].append(box)

    result = []

    for bboxes in dict.values():
        while len(bboxes) > 0:
            chosen_box = bboxes.pop()
            bboxes = [box for box in bboxes if
                      intersection_over_union(
                          torch.tensor(box[2:]),
                          torch.tensor(chosen_box[2:]),
                          box_format=box_format
                      ) < iou_threshold]
            result.append(chosen_box)

    return result


def map(
        pred_boxes: list[list[int | float]],
        true_boxes: list[list[int | float]],
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=20,
        epsilon=1e-6
) -> float:
    """
    mAP 값을 계산하여 반환한다.

    Parameters:
        pred_boxes (list): [train_idx, class_prediction, prob_score, x1, y1, x2, y2]로 이루어진 list
        true_boxes (list): [train_idx, class_idx, prob_score(있으면 1, 없으면 0), x1, y1, x2, y2]로 이루어진 list
        iou_threshold (float): TP, FP를 결정하는 기준값
        box_format (str): bounding box 형식, "midpoint" 또는 "corners"
        num_classes (int): 클래스 수

    Returns:
        float: iou_threshold 값을 기준으로 모든 클래스에 대한 평균 mAP를 반환한다.
    """
    average_precisions = []

    # class 별로 분류
    detections = [[] for _ in range(num_classes)]
    for detection in pred_boxes:
        detections[detection[1]].append(detection)

    ground_truths = [[] for _ in range(num_classes)]
    for true_box in true_boxes:
        ground_truths[true_box[1]].append(true_box)

    for c in range(num_classes):
        if not ground_truths[c]:
            continue

        # train_idx 별로 몇 개인지 셈
        amount_bboxes = Counter([gt[0] for gt in ground_truths[c]])

        # ground_truth 판정 여부 저장
        is_evaluated = defaultdict(bool)
        for key, val in amount_bboxes.items():
            is_evaluated[key] = [False] * val

        # confidence 기준 내림차순으로 정렬
        class_detections = sorted(detections[c], key=lambda x: x[2], reverse=True)

        # TP, FP 여부 저장
        TP = torch.zeros((len(class_detections)))
        FP = torch.zeros((len(class_detections)))

        data_ground_truths = defaultdict(list)
        for bbox in ground_truths[c]:
            data_ground_truths[bbox[0]].append(bbox)

        for detection_idx, detection in enumerate(class_detections):
            data_ground_truth = data_ground_truths[detection[0]]

            best_iou = 0
            best_gt_idx = 0

            for data_idx, gt in enumerate(data_ground_truth):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if best_iou < iou:
                    best_iou = iou
                    best_gt_idx = data_idx

            # iou_threshold 초과이고 이미 판정되지 않았으면(정렬되어 있어 이전의 결과가 최선임을 보장) TP, 아니면 FP
            if best_iou > iou_threshold and is_evaluated[detection[0]][best_gt_idx] == 0:
                TP[detection_idx] = 1
                is_evaluated[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # TP, FP의 누적합
        TP = torch.cumsum(TP, dim=0)
        FP = torch.cumsum(FP, dim=0)

        # recall = TP / (TP + FN) = TP / all ground truths
        # precision = TP / (TP + FP) = TP / all detections
        num_ground_truths = len(ground_truths[c])
        recalls = TP / (num_ground_truths + epsilon)
        precisions = TP / (TP + FP + epsilon)

        # (0, 1) 추가
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        # PR 곡선의 적분값 = AP
        average_precisions.append(torch.trapz(precisions, recalls))

    # AP의 평균 = mAP
    return sum(average_precisions) / len(average_precisions)


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        pred_format="cells",
        box_format="midpoint",
        device="mps",
        S=7,
        C=20
):
    all_pred_boxes = []
    all_true_boxes = []

    loop = tqdm(loader, leave=True)
    loop.set_description(f"Evaluating bboxes")

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loop):

        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels, S, C)
        bboxes = cellboxes_to_boxes(predictions, S, C)

        for idx in range(batch_size):
            nms_boxes = nms(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            # if idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7, C=11):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 10)
    bboxes1 = predictions[..., C + 1:C + 5]
    bboxes2 = predictions[..., C + 6:C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds


def cellboxes_to_boxes(out, S=7, C=11):
    converted_pred = convert_cellboxes(out, S, C).reshape(out.shape[0], S * S, -1)
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            items = [x.item() for x in converted_pred[ex_idx, bbox_idx, :]]
            items[0] = int(items[0])
            bboxes.append(items)
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("=> Loaded checkpoint")
