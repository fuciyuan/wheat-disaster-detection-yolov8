# 数据集结构

## 1. 数据集概述

本项目使用的数据集是关于小麦灾害检测的，包含 5 种灾害类型：

1. rust: 锈病
2. powdery_mildew: 白粉病
3. aphids: 蚜虫
4. grasshopper: 蝗虫
5. drought: 干旱

## 2. 数据集结构

### 2.1 目录结构

数据集应该按照 YOLO 格式组织：

```
data/wheat_disaster_dataset/
├── images/
│   ├── train/          # 训练集图像
│   ├── val/            # 验证集图像
│   └── test/           # 测试集图像
└── labels/
    ├── train/          # 训练集标签文件
    ├── val/            # 验证集标签文件
    └── test/           # 测试集标签文件
```

### 2.2 标签格式

每个标签文件包含如下格式：

```
class_id x_center y_center width height
```

例如：
```
0 0.1 0.2 0.3 0.4
1 0.5 0.6 0.7 0.8
```

### 2.3 类别配置

在数据集配置文件 `src/data/wheat_disaster.yaml` 中，包含以下内容：

```yaml
path: data/wheat_disaster_dataset
train: images/train
val: images/val
test: images/test
nc: 5
names:
  0: rust
  1: powdery_mildew
  2: aphids
  3: grasshopper
  4: drought
```

### 2.4 图像格式

图像可以是 JPG、PNG、BMP 等常见格式。

### 2.5 标签文件命名

标签文件名称与对应的图像文件名称相同，但扩展名为 `.txt`。例如，如果图像文件为 `test.jpg`，对应的标签文件为 `test.txt`。

## 3. 数据集划分

### 3.1 划分比例

建议的划分比例：

- 训练集: 70%
- 验证集: 20%
- 测试集: 10%

### 3.2 划分方式

可以使用 Python 脚本自动划分数据集：

```python
import os
import shutil
import random

input_dir = './data/wheat_disaster_dataset'
output_dir = './data/wheat_disaster_dataset_split'

os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

image_files = os.listdir(os.path.join(input_dir, 'images'))
random.shuffle(image_files)

train_files = image_files[:int(0.7 * len(image_files))]
val_files = image_files[int(0.7 * len(image_files)):int(0.9 * len(image_files))]
test_files = image_files[int(0.9 * len(image_files)):]

for file in train_files:
    shutil.copy(os.path.join(input_dir, 'images', file), os.path.join(output_dir, 'images', 'train', file))
    label_file = file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(input_dir, 'labels', label_file)):
        shutil.copy(os.path.join(input_dir, 'labels', label_file), os.path.join(output_dir, 'labels', 'train', label_file))

for file in val_files:
    shutil.copy(os.path.join(input_dir, 'images', file), os.path.join(output_dir, 'images', 'val', file))
    label_file = file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(input_dir, 'labels', label_file)):
        shutil.copy(os.path.join(input_dir, 'labels', label_file), os.path.join(output_dir, 'labels', 'val', label_file))

for file in test_files:
    shutil.copy(os.path.join(input_dir, 'images', file), os.path.join(output_dir, 'images', 'test', file))
    label_file = file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(input_dir, 'labels', label_file)):
        shutil.copy(os.path.join(input_dir, 'labels', label_file), os.path.join(output_dir, 'labels', 'test', label_file))

print('数据集划分完成。')
```

## 4. 数据集准备

### 4.1 数据集下载

可以通过以下方式下载数据集：

- 公开数据集下载
- 收集公开的数据，如 Kaggle 数据集中的小麦灾害图像
- 自行采集小麦灾害图像

### 4.2 数据采集

可以通过以下方式采集图像：

- 使用无人机拍摄小麦田图像
- 使用数码相机在小麦田拍摄图像
- 使用公开数据集（如 Kaggle、GitHub 等）

### 4.3 数据标注

可以使用以下工具进行标注：

- [LabelImg](https://github.com/tzutalin/labelImg)
- [LabelBox](https://labelbox.com/)
- [CVAT](https://github.com/opencv/cvat)

### 4.4 数据清洗

数据清洗包括：

- 删除损坏的图像
- 修正错误的标签
- 删除模糊的图像
- 补充缺失的标签

### 4.5 数据增强

可以使用以下方法进行数据增强：

```python
import albumentations as A
import cv2
import numpy as np

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.1),
    A.RandomRotate90(p=0.5),
])

image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transformed = transform(image=image)
transformed_image = transformed['image']
```

### 4.6 数据转换

可以使用以下方法将数据转换为 YOLO 格式：

```python
import os
import xml.etree.ElementTree as ET

def convert_voc_to_yolo(voc_dir, yolo_dir, classes):
    os.makedirs(yolo_dir, exist_ok=True)
    for xml_file in os.listdir(voc_dir):
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()
        
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        
        yolo_file = os.path.join(yolo_dir, os.path.splitext(filename)[0] + '.txt')
        with open(yolo_file, 'w', encoding='utf-8') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                width_norm = (xmax - xmin) / width
                height_norm = (ymax - ymin) / height
                
                f.write(f'{cls_id} {x_center} {y_center} {width_norm} {height_norm}\n')

# 使用示例
classes = ['rust', 'powdery_mildew', 'aphids', 'grasshopper', 'drought']
convert_voc_to_yolo('./data/voc_labels', './data/yolo_labels', classes)
```

## 5. 数据集质量评估

### 5.1 评估指标

可以使用以下指标评估数据集质量：

1. 图像清晰度
2. 标签质量
3. 类别分布
4. 边界框质量

### 5.2 评估工具

可以使用以下工具评估数据集质量：

1. [CVAT](https://github.com/opencv/cvat) - 可视化和评估数据集
2. [LabelStudio](https://labelstud.io/) - 数据标注和质量评估

### 5.3 评估方法

可以使用以下方法评估数据集质量：

```python
import os
import json

def evaluate_dataset(dataset_dir):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    total_images = 0
    total_labels = 0
    invalid_labels = 0
    
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(images_dir, split)
        split_labels_dir = os.path.join(labels_dir, split)
        
        if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
            continue
        
        for image_file in os.listdir(split_images_dir):
            total_images += 1
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(split_labels_dir, label_file)
            
            if not os.path.exists(label_path):
                invalid_labels += 1
                continue
            
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
            
            total_labels += len(labels)
            
            for label in labels:
                parts = label.split()
                if len(parts) != 5:
                    invalid_labels += 1
                else:
                    try:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if not (0 ≤ x_center ≤ 1 and 0 ≤ y_center ≤ 1 and 0 < width ≤ 1 and 0 < height ≤ 1):
                            invalid_labels += 1
                    except:
                        invalid_labels += 1
    
    report = {
        'total_images': total_images,
        'total_labels': total_labels,
        'invalid_labels': invalid_labels,
        'label_density': total_labels / total_images if total_images > 0 else 0,
        'invalid_rate': invalid_labels / (total_images + total_labels) if (total_images + total_labels) > 0 else 0,
    }
    
    return report

# 使用示例
report = evaluate_dataset('data/wheat_disaster_dataset')
print(json.dumps(report, indent=2))
```

## 6. 数据集版本管理

### 6.1 版本控制

可以使用 Git 进行数据集版本控制：

```bash
git lfs track '*.jpg'
git lfs track '*.png'
git add .gitattributes
git add data/
git commit -m 'Add dataset v1.0'
git push origin main
```

### 6.2 数据集版本

可以为数据集添加版本信息：

```yaml
# dataset.yaml
version: 1.0
name: Wheat Disaster Detection Dataset
description: A dataset for detecting wheat diseases and pests
classes:
  - rust
  - powdery_mildew
  - aphids
  - grasshopper
  - drought
images:
  train: data/train/images
  val: data/val/images
  test: data/test/images
labels:
  train: data/train/labels
  val: data/val/labels
  test: data/test/labels
annotations:
  format: yolo
  version: 1.0
```

### 6.3 数据集元数据

可以为数据集添加元数据：

```yaml
# metadata.yaml
name: Wheat Disaster Detection Dataset
description: A dataset for detecting wheat diseases and pests
version: 1.0
creator: John Doe
created: 2024-01-01
updated: 2024-01-01
license: CC BY 4.0
size: 1000 images
classes:
  - name: rust
    description: Wheat rust disease
    examples: 200
  - name: powdery_mildew
    description: Wheat powdery mildew disease
    examples: 200
  - name: aphids
    description: Wheat aphids pest
    examples: 200
  - name: grasshopper
    description: Wheat grasshopper pest
    examples: 200
  - name: drought
    description: Wheat drought stress
    examples: 200
```

## 7. 数据集使用

### 7.1 加载数据集

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

class WheatDisasterDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
        self.label_files = [self.labels_dir / (img_file.stem + '.txt') for img_file in self.image_files]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id, x_center, y_center, width, height = map(float, parts)
                    labels.append(int(cls_id))
                    
                    # 转换为绝对坐标
                    h, w = image.shape[:2]
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    boxes.append([x1, y1, x2, y2])
        
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        return image, boxes, labels

# 使用示例
dataset = WheatDisasterDataset('data/wheat_disaster_dataset/images/train', 'data/wheat_disaster_dataset/labels/train')
print(f'数据集大小: {len(dataset)}')
image, boxes, labels = dataset[0]
print(f'图像形状: {image.shape}')
print(f'边界框: {boxes}')
print(f'标签: {labels}')
```

### 7.2 数据加载

```python
import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    images = []
    boxes_list = []
    labels_list = []
    
    for image, boxes, labels in batch:
        images.append(image)
        boxes_list.append(boxes)
        labels_list.append(labels)
    
    images = torch.stack(images)
    
    return images, boxes_list, labels_list

# 数据加载器
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

for batch_idx, (images, boxes_list, labels_list) in enumerate(loader):
    print(f'批次: {batch_idx + 1}')
    print(f'图像形状: {images.shape}')
    print(f'边界框数量: {len(boxes_list)}')
    print(f'标签数量: {len(labels_list)}')
    break
```

## 8. 数据集部署

### 8.1 数据集部署到模型

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='src/data/wheat_disaster.yaml', epochs=100)
```

### 8.2 数据集部署到移动设备

可以使用以下方法部署数据集到移动设备：

```python
# 使用 CoreML 部署
import coremltools as ct

model = YOLO('yolov8n.pt')
model.export(format='coreml')
```

### 8.3 数据集部署到服务器

可以使用以下方法部署数据集到服务器：

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/dataset', methods=['GET'])
def get_dataset():
    return jsonify({
        'name': 'Wheat Disaster Detection Dataset',
        'version': '1.0',
        'classes': ['rust', 'powdery_mildew', 'aphids', 'grasshopper', 'drought'],
        'total_images': 1000,
        'total_labels': 2000,
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## 9. 下一步

- 阅读 [快速开始](quick-start.md) 了解基本使用方法
- 阅读 [训练指南](training.md) 了解更多训练技巧
- 阅读 [API 文档](api.md) 了解更多接口
- 阅读 [配置说明](configuration.md) 了解更多配置选项
- 阅读 [FAQ](faq.md) 了解常见问题

## 10. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)