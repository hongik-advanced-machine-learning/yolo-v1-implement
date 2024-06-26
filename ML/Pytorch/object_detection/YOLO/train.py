import os
import time
import yaml

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import cv2

from model import YOLOv1
from dataset import CustomDataset
from utils import *
from loss import YoloLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

torch.manual_seed(config["pytorch"]["seed"])

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

LEARNING_RATE = config["param"]["learning_rate"]
BATCH_SIZE = config["param"]["batch_size"]
WEIGHT_DECAY = config["param"]["weight_decay"]
EPOCHS = config["param"]["epochs"]

NUM_WORKERS = config["pytorch"]["num_workers"]
PIN_MEMORY = config["pytorch"]["pin_memory"]
SAVE_MODEL = config["pytorch"]["save_model"]
LOAD_MODEL_FILE = config["pytorch"]["model_path"]

DATASET = config["dataset"]["name"]
NUM_CLASS = config["dataset"]["num_class"]

# Data augmentation with horizontal and vertical flips
train_transforms_original = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_hflip = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.HorizontalFlip(p=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_vflip = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.VerticalFlip(p=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_Rotation = A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Resize(448, 448),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_RandomScale = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.RandomScale(scale_limit=0.2, p=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_RandomMove = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.0, rotate_limit=0, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transforms_RandomColor = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0.5, val_shift_limit=0.5, p=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

test_transforms = A.Compose([
    A.Normalize(mean=0.0, std=1.0),
    A.Resize(448, 448),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_dataloader():
    original_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_original,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    hflip_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_hflip,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    vflip_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_vflip,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    Rotation_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_Rotation,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    RandomScale_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_RandomScale,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    RandomMove_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_RandomMove,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    RandomColor_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=train_transforms_RandomColor,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS,
    )

    train_dataset = ConcatDataset([original_dataset, RandomScale_dataset, RandomColor_dataset, hflip_dataset, RandomMove_dataset, Rotation_dataset])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_dataset = CustomDataset(
        os.path.join("data", DATASET, "test.csv"),
        transform=test_transforms,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
        C=NUM_CLASS
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, test_loader


def load_model_optimizer_scheduler():
    model = YOLOv1(split_size=7, num_boxes=2, num_classes=NUM_CLASS).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=0)

    try:
        load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=DEVICE), model, optimizer)
    except FileNotFoundError:
        pass

    return model, optimizer, scheduler


def train_fn(model, optimizer, loss_fn, loop):
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        optimizer.zero_grad()
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean train loss: {sum(mean_loss) / len(mean_loss):.4f}")


def test_fn(model, scheduler, loss_fn, loop):
    with torch.no_grad():
        model.eval()

        mean_loss = []

        for batch_idx, (x, y) in enumerate(loop):
            optimizer.zero_grad()
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = loss_fn(out, y)
            mean_loss.append(loss.item())

            # update progress bar
            loop.set_postfix(loss=loss.item())

        mean = sum(mean_loss) / len(mean_loss)
        scheduler.step(mean)
        print(f"Mean test loss: {mean:.4f}")

        model.train()


def print_map(model, train_loader, test_loader):
    # pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS)
    # mean_avg_prec = mean_average_precision(
    #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    # )
    # print(f"Train mAP: {mean_avg_prec:.5g}")

    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS,
                                          device=DEVICE)
    mean_avg_prec = map(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    )
    print(f"Test mAP: {mean_avg_prec:.5g}")


def main(model, optimizer, scheduler):
    train_loader, test_loader = load_dataloader()

    loss_fn = YoloLoss(S=7, B=2, C=NUM_CLASS).to(DEVICE)

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}, LR: {optimizer.param_groups[0]["lr"]}")

        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"[Train] Epoch {epoch + 1}")
        train_fn(model, optimizer, loss_fn, train_loop)

        time.sleep(5)

        test_loop = tqdm(test_loader, leave=True)
        test_loop.set_description(f"[Test] Epoch {epoch + 1}")
        test_fn(model, scheduler, loss_fn, test_loop)

        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(60)

        print_map(model, train_loader, test_loader)


if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    model, optimizer, scheduler = load_model_optimizer_scheduler()
    main(model, optimizer, scheduler)
