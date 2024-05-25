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
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1
NUM_WORKERS = 8
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = False
LOAD_MODEL_FILE = "save/voc.dropout.tar"

DATASET = "VOC"
NUM_CLASS = 20

transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])


def load_dataloader():
    train_dataset = CustomDataset(
        os.path.join("data", DATASET, "train.csv"),
        transform=transform,
        img_dir=os.path.join("data", DATASET, "images"),
        label_dir=os.path.join("data", DATASET, "labels"),
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
    model = Yolov1(split_size=7, num_boxes=2, num_classes=NUM_CLASS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

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
    pred_boxes, target_boxes = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS)
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    )
    print(f"Train mAP: {mean_avg_prec:.5g}")

    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, threshold=0.4, S=7, C=NUM_CLASS,
                                          device=DEVICE)
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=NUM_CLASS
    )
    print(f"Test mAP: {mean_avg_prec:.5g}")


def main(model, optimizer, scheduler):
    train_loader, test_loader = load_dataloader()

    loss_fn = YoloLoss(S=7, B=2, C=NUM_CLASS).to(DEVICE)

    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch + 1}, LR: {optimizer.param_groups[0]['lr']}")

        train_loop = tqdm(train_loader, leave=True)
        train_loop.set_description(f"[Train] Epoch {epoch + 1}")
        train_fn(model, optimizer, loss_fn, train_loop)

        time.sleep(5)

        test_loop = tqdm(train_loader, leave=True)
        test_loop.set_description(f"[Test] Epoch {epoch + 1}")
        test_fn(model, scheduler, loss_fn, test_loop)

        if SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            time.sleep(60)

    time.sleep(5)
    print_map(model, train_loader, test_loader)


if __name__ == "__main__":
    print(f"Using {DEVICE} device")
    model, optimizer, scheduler = load_model_optimizer_scheduler()
    main(model, optimizer, scheduler)
