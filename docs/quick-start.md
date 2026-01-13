# Quick Start Guide

## 1. 项目初始化

### 1.1 克隆项目

```bash
git clone https://github.com/your-username/wheat-disaster-detection-yolov8.git
cd wheat-disaster-detection-yolov8
```

### 1.2 安装uv

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.3 安装依赖

```bash
uv sync
```

### 1.4 验证安装

```bash
python test.py
```

如果看到以下信息，表示安装成功：
```
小麦灾害检测系统初始化成功！

可以使用以下命令运行：
  python src/main.py detect --image path/to/image.jpg
  python src/main.py train --data data/wheat_disaster.yaml --epochs 100
```

## 2. 数据集准备

### 2.1 数据集下载

可以从以下地方获取小麦灾害数据集：
1. 自建数据集
2. 公开数据集（如Plant Village, Kaggle Wheat Disease）
3. 数据标注平台导出

### 2.2 数据集结构

```
data/
├── wheat_disaster.yaml    # 数据集配置文件
├── train/
│   ├── images/
│   │   ├── img0001.jpg
│   │   └── img0002.jpg
│   └── labels/
│       ├── img0001.txt
│       └── img0002.txt
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    └── labels/
```

### 2.3 配置文件

修改 `src/config/config.yaml`：

```yaml
model:
  name: "yolov8n"
  imgsz: 640
  conf: 0.25
  iou: 0.45

train:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  save_dir: "runs/train"

dataset:
  path: "data"
  train: "train/images"
  val: "val/images"
  test: "test/images"
  nc: 5
  names: ['rust', 'powdery_mildew', 'aphid', 'wheat_blast', 'healthy']
```

## 3. 模型训练

### 3.1 预训练模型

下载YOLOv8n预训练模型：
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3.2 训练模型

```bash
python src/main.py train --data data/wheat_disaster.yaml --epochs 100
```

训练完成后，结果会保存在 `runs/train` 目录

## 4. 模型检测

### 4.1 单张图片检测

```bash
python src/main.py detect --image examples/test.jpg
```

### 4.2 批量检测

```bash
python src/main.py detect --images examples/
```

### 4.3 视频检测

```bash
python src/main.py detect --video examples/test.mp4
```

### 4.4 实时检测

```bash
python src/main.py detect --camera
```

## 5. 模型评估

```bash
python src/main.py val --data data/wheat_disaster.yaml --weights runs/train/exp/weights/best.pt
```

## 6. 常见问题

### 问题1：uv sync 失败

可能的解决方法：
1. 网络问题：检查网络连接
2. 代理问题：如果使用代理，请设置环境变量

### 问题2：GPU加速问题

如果需要GPU加速，请确保安装了CUDA：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 7. 性能优化

1. 使用更大的模型（yolov8s, yolov8m, yolov8l, yolov8x）
2. 调整batch size和epochs参数
3. 增加数据增强
4. 使用FP16半精度训练

## 8. 部署

### 8.1 导出模型

```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

### 8.2 部署方式

1. Flask/FastAPI部署
2. Tensoflow Serving部署
3. ONNX Runtime部署
4. TensorRT部署

