import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=2, C=11, transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        class_labels = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = map(float, label.strip().split())
                x_min = x - width / 2
                y_min = y - height / 2
                x_max = x + width / 2
                y_max = y + height / 2
                boxes.append([x_min, y_min, x_max, y_max])
                class_labels.append(class_label)
        boxes = np.array(boxes)
        class_labels = np.array(class_labels)

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            class_labels = transformed['class_labels']

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box, class_label in zip(boxes, class_labels):
            x_min, y_min, x_max, y_max = box
            class_label = int(class_label)
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Ensure x, y are within bounds
            x = min(0.9999, max(0, x))
            y = min(0.9999, max(0, y))

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                label_matrix[i, j, class_label] = 1

        return image.float(), label_matrix
