#실험용
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import CustomDataset
import yaml

cmap = ['r', 'g', 'b', 'k', 'w', 'c', 'm', 'y', '#007bff', '#d62728',
          '#28a745', '#ffc107', '#dc3545', '#fd7e14', '#198754', '#000080',
          '#6600cc', '#808080', '#008080', '#ffa65c'] # 클래스 별 색 지정

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
  img_index = [25,19,28]
  ####################################################
  NUM_CLASS = 20
  COLUMNS = len(img_index)

  train_dataset = CustomDataset(
          config['train_csv_path'],
          transform= transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),]),
          img_dir=config['img_dir'],
          label_dir=['label_dir'],
          C=NUM_CLASS,
      )

  ###########################타겟 그리드 셀 시각화#########################
  fig, ax = plt.subplots( 2,COLUMNS, figsize=((10 * COLUMNS, 20)))
  fig


  for t in range(COLUMNS):
    image, label_matrix = train_dataset[img_index[t]]

    im = image.permute(1,2,0)

    ax[0][t].imshow(im)

    for i in range(1,7):
      ax[0][t].hlines(i*64, 0, 447, color='black', linestyles='solid', linewidth= 1)
      ax[0][t].vlines(i*64, 0, 447, color='black', linestyles='solid', linewidth= 1)

    for i in range(7):
      for j in range(7):
        if (label_matrix[i,j, NUM_CLASS] == 1):
          # print("selected!", i, j)
          index = 0

          for k,value in enumerate(label_matrix[i,j,0:20]):
            if value == 1:
              # print("index", index)
              index = k
              break

          ax[0][t].add_patch(
            patches.Rectangle(
                (64* j, 64* i),                   # (x, y)
                63.5, 63.5,                     # width, height
                edgecolor = 'black',
                facecolor = cmap[index],
                alpha = 0.6,
                fill=True,
            )
          )

    ######################################################################################

    ###########################중심좌표 변환 이미지 코드 + 바운딩박스 시각화###############################################
    for i in range(7):
      for j in range(7):
        if (label_matrix[i,j, NUM_CLASS] == 1):
          index = 0

          for k,value in enumerate(label_matrix[i,j,0:20]):
            if value == 1:
              index = k
              break



          x = (( j + label_matrix[i,j,NUM_CLASS+1]) / 7) * 448 - label_matrix[i,j,NUM_CLASS + 3] * 64/2
          y = (( i + label_matrix[i,j,NUM_CLASS+2]) / 7) * 448 - label_matrix[i,j,NUM_CLASS + 4] * 64/2
          ax[1][t].imshow(im)
          # Create a Rectangle patch
          rect = patches.Rectangle((x, y), label_matrix[i,j,NUM_CLASS + 3] * 64, label_matrix[i,j,NUM_CLASS + 4] * 64 , linewidth=4, edgecolor=cmap[index], facecolor='none')
          # Add the patch to the Axes
          ax[1][t].add_patch(rect)

          # rect = patches.Rectangle((x, y -20), 75, 20, linewidth=4, edgecolor=cmap[index], facecolor=cmap[index])
          # ax[1][t].add_patch(rect)
          #text print
          ax[1][t].text(x,y, name[index], color='white', fontsize=20, bbox=dict(facecolor=cmap[index]))
          # plt.scatter(label_matrix[i,j,], y , color=cmap[index])

  plt.show()

#############################################################################