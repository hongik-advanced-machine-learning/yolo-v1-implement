# 실험용
import os
import yaml

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import CustomDataset

DATASET = "VOC"

cmap = ['r', 'g', 'b', 'k', 'w', 'c', 'm', 'y', '#007bff', '#d62728',
        '#28a745', '#ffc107', '#dc3545', '#fd7e14', '#198754', '#000080',
        '#6600cc', '#808080', '#008080', '#ffa65c']  # 클래스 별 색 지정

name = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Diningtable",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Pottedplant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
]

if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    ####################################여기 img_index조작해서 시각화할 이미지 넘버 넣기 주의! (2개이상부터 동작)
    img_index = [0, 1, 2]
    ####################################################
    NUM_CLASS = 20
    COLUMNS = len(img_index)

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

    test_transforms = A.Compose([
        A.Normalize(mean=0.0, std=1.0),
        A.Resize(448, 448),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

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

    ###########################타겟 그리드 셀 시각화#########################
    fig, ax = plt.subplots(2, COLUMNS, figsize=((10 * COLUMNS, 20)))
    fig

    for t in range(COLUMNS):
        image, label_matrix = hflip_dataset[img_index[t]]

        im = image.permute(1, 2, 0)

        ax[0][t].imshow(im)

        for i in range(1, 7):
            ax[0][t].hlines(i * 64, 0, 447, color='black', linestyles='solid', linewidth=1)
            ax[0][t].vlines(i * 64, 0, 447, color='black', linestyles='solid', linewidth=1)

        for i in range(7):
            for j in range(7):
                if (label_matrix[i, j, NUM_CLASS] == 1):
                    # print("selected!", i, j)
                    index = 0

                    for k, value in enumerate(label_matrix[i, j, 0:20]):
                        if value == 1:
                            # print("index", index)
                            index = k
                            break

                    ax[0][t].add_patch(
                        patches.Rectangle(
                            (64 * j, 64 * i),  # (x, y)
                            63.5, 63.5,  # width, height
                            edgecolor='black',
                            facecolor=cmap[index],
                            alpha=0.6,
                            fill=True,
                        )
                    )

        ######################################################################################

        ###########################중심좌표 변환 이미지 코드 + 바운딩박스 시각화###############################################
        for i in range(7):
            for j in range(7):
                if (label_matrix[i, j, NUM_CLASS] == 1):
                    index = 0

                    for k, value in enumerate(label_matrix[i, j, 0:20]):
                        if value == 1:
                            index = k
                            break

                    x = ((j + label_matrix[i, j, NUM_CLASS + 1]) / 7) * 448 - label_matrix[i, j, NUM_CLASS + 3] * 64 / 2
                    y = ((i + label_matrix[i, j, NUM_CLASS + 2]) / 7) * 448 - label_matrix[i, j, NUM_CLASS + 4] * 64 / 2
                    ax[1][t].imshow(im)
                    # Create a Rectangle patch
                    rect = patches.Rectangle((x, y), label_matrix[i, j, NUM_CLASS + 3] * 64,
                                             label_matrix[i, j, NUM_CLASS + 4] * 64, linewidth=4, edgecolor=cmap[index],
                                             facecolor='none')
                    # Add the patch to the Axes
                    ax[1][t].add_patch(rect)

                    # rect = patches.Rectangle((x, y -20), 75, 20, linewidth=4, edgecolor=cmap[index], facecolor=cmap[index])
                    # ax[1][t].add_patch(rect)
                    # text print
                    ax[1][t].text(x, y, name[index], color='white', fontsize=20, bbox=dict(facecolor=cmap[index]))
                    # plt.scatter(label_matrix[i,j,], y , color=cmap[index])

    plt.show()

#############################################################################
