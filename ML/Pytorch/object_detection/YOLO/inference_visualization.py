from model import YOLOv1
from dataset import CustomDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import get_bboxes, mean_average_precision
import torch
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

#모델 가중치 파일 경로 설정
LOAD_MODEL_FILE = config['pytorch']['model_path']#"/content/drive/MyDrive/dataset/voc.tar"
DEVICE = 'cuda'
NUM_CLASS = 20

# model 생성
model = YOLOv1(split_size=7, num_boxes=2, num_classes=NUM_CLASS).to(DEVICE)

model.load_state_dict(torch.load(LOAD_MODEL_FILE)["state_dict"])


#test dataset 생성
test_dataset = CustomDataset(
  config['dataset']['test_path'],
  transform=transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()]),
  img_dir=config['dataset']['img_dir'],
  label_dir=config['dataset']['label_dir'],
  C=NUM_CLASS
)

# test dataset loader
test_loader = DataLoader(
  dataset=test_dataset,
  batch_size=1,
  num_workers=2,
  pin_memory=False,
  shuffle=False,
  drop_last=True,
)


# 학습된 model로 test dataset(== train_dataset)의 prediction box와 target box 생성
pred_boxes, target_boxes = get_bboxes(
    test_loader, model, iou_threshold=0.5, threshold=0.4
)

print("pre_boxes: ", torch.Tensor(pred_boxes).shape)
print("target_boxes: ", torch.Tensor(target_boxes).shape)
# model이 얼마나 정확히 예측하였는지 mAP계산
mean_avg_prec = mean_average_precision(
    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
)

# Test data의 mAP 계산
print(f"Test mAP: {mean_avg_prec}")