import os
import time

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from model import Yolov1
from dataset import CustomDataset
from utils import *
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
LEARNING_RATE = 1e-6
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-5
EPOCHS = 80
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
LOAD_MODEL_FILE = "save/voc.dropout.tar"

DATASET = "VOC"
TRAIN = ""
TEST = ""
NUM_CLASS = 20

transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def lr_lambda(epoch):
    # if epoch < 10:
    #     return (epoch + 1) / 10
    if epoch < 20:
        return 1
    elif epoch < 50:
        return 0.1
    else:
        return 0.01


def train_fn(model, optimizer, scheduler, loss_fn, loop):
    loss_sum = 0.
    length = 0

    for batch_idx, (x, y) in enumerate(loop):
        optimizer.zero_grad()
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        loss = loss_fn(out, y)
        loss_sum += loss.item()
        length += 1

        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss: {loss_sum / length}")
    time.sleep(1)


def model_optimizer_scheduler():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lr_lambda)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    return model, optimizer, scheduler


def main(model, optimizer, scheduler):
    train_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=transform,
        img_dir=os.path.join("data", DATASET, TRAIN, "images"),
        label_dir=os.path.join("data", DATASET, TRAIN, "labels"),
        C=NUM_CLASS,
    )

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
        transform=transform,
        img_dir=os.path.join("data", DATASET, TEST, "images"),
        label_dir=os.path.join("data", DATASET, TEST, "labels"),
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

    loss_fn = YoloLoss(S=7, B=2, C=NUM_CLASS).to(DEVICE)

    for epoch in range(55, EPOCHS):
        print(f"Epoch: {epoch + 1}, LR: {optimizer.param_groups[0]['lr']}")

        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch {epoch + 1}")

        train_fn(model, optimizer, scheduler, loss_fn, loop)

        pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS)
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
        )
        print(f"Test mAP: {mean_avg_prec}")

        scheduler.step()

        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(60)


def test(model):
    train_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=transform,
        img_dir=os.path.join("data", DATASET, TRAIN, "images"),
        label_dir=os.path.join("data", DATASET, TRAIN, "labels"),
        C=NUM_CLASS,
    )

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
        transform=transform,
        img_dir=os.path.join("data", DATASET, TEST, "images"),
        label_dir=os.path.join("data", DATASET, TEST, "labels"),
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

    pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS)
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    )
    print(f"Train mAP: {mean_avg_prec}")

    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS)
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    )
    print(f"Test mAP: {mean_avg_prec}")


if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    model, optimizer, scheduler = model_optimizer_scheduler()
    # main(model, optimizer, scheduler)
    test(model)
