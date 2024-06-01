"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


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
                boxes.append([x, y, width, height])
                class_labels.append(int(class_label))

        boxes = np.array(boxes)
        class_labels = np.array(class_labels)

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path))
        boxes = np.array(boxes)

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'])
            class_labels = transformed['class_labels']
            # image = self.transform(image)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box, class_label in zip(boxes, class_labels):
            x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, self.C+1:self.C+5] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
