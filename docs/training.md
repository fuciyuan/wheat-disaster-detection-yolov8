# 训练指南

## 1. 数据集准备

### 1.1 数据集下载

可以从以下公开资源获取小麦灾害数据集：
- [Plant Village Dataset](https://www.kaggle.com/emmarex/plantdisease) - 包含植物病害数据
- [Kaggle Wheat Dataset](https://www.kaggle.com/c/global-wheat-detection) - 小麦病害数据集
- [Agricultural Disease Dataset](https://www.kaggle.com/koryakinp/wheat-disease-detection)

### 1.2 数据标注

使用标注工具（如LabelImg, LabelStudio, CVAT）对数据集进行标注，标签格式采用YOLO格式（TXT文件）。

**标注文件格式要求：**
- 每行表示一个标注目标
- 格式：`class_id x_center y_center width height`
- 所有坐标相对于图像宽度和高度进行了归一化（值在0到1之间）

### 1.3 数据集结构

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

## 2. 训练配置

### 2.1 模型选择

YOLOv8提供了多种模型大小：

| 模型 | 参数大小 | 速度 | 精度 |
| --- | --- | --- | --- |
| yolov8n | 3.2M | 最快 | 较低 |
| yolov8s | 11.2M | 快 | 高 |
| yolov8m | 25.9M | 中 | 较高 |
| yolov8l | 43.7M | 慢 | 很高 |
| yolov8x | 68.2M | 最慢 | 最高 |

### 2.2 参数调整

修改 `config/config.yaml`：

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

### 2.3 数据增强

YOLOv8支持多种数据增强方法（在训练期间自动应用）：
- 随机裁剪
- 随机翻转
- 随机旋转
- 色彩增强
- 马赛克增强

## 3. 开始训练

### 3.1 训练命令

```bash
python src/main.py train --data data/wheat_disaster.yaml --epochs 100
```

**完整训练命令示例：**

```bash
python src/main.py train \
  --data data/wheat_disaster.yaml \
  --epochs 100 \
  --batch_size 16 \
  --imgsz 640 \
  --name wheat_disaster_yolov8
```

### 3.2 训练过程

训练时将输出以下信息：
- 当前训练轮数和总轮数
- 损失值（box_loss, cls_loss, dfl_loss）
- 评估结果（mAP50, mAP50-95, precision, recall）
- 训练耗时

## 4. 训练监控

### 4.1 TensorBoard监控

训练过程中可以使用TensorBoard监控训练过程：

```bash
# 启动TensorBoard服务器
tensorboard --logdir runs/train
```

在浏览器中打开 `http://localhost:6006` 查看训练曲线和指标。

### 4.2 Weights & Biases

使用Weights & Biases进行更详细的训练监控：

```bash
# 安装wandb
uv pip install wandb

# 登录（需要先注册账号）
wandb login

# 训练
python src/main.py train --data data/wheat_disaster.yaml --epochs 100 --wandb
```

## 5. 训练结果

### 5.1 结果目录

训练完成后，结果将保存在 `runs/train` 目录下：

```
runs/train/exp/
├── weights/
│   ├── best.pt    # 验证集上表现最好的权重
│   └── last.pt    # 最后一次迭代的权重
├── val_batch0_pred.jpg  # 预测结果示例
├── results.csv  # 训练结果汇总
├── confusion_matrix.png  # 混淆矩阵
└── PR_curve.png  # PR曲线
```

### 5.2 评估指标

常见评估指标及其含义：
- **mAP (mean Average Precision)**: 检测性能的主要指标
- **mAP50**: IoU阈值为0.50时的mAP
- **mAP50-95**: IoU阈值从0.50到0.95的平均mAP
- **precision**: 预测为阳性且实际为阳性的比例
- **recall**: 实际为阳性且被预测为阳性的比例
- **F1**: precision和recall的调和平均数

## 6. 调优策略

### 6.1 学习率调优

#### 学习率预热

```yaml
train:
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
```

#### 学习率衰减

```yaml
train:
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
```

### 6.2 数据增强策略

```yaml
train:
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
  copy_paste: 0.0
```

### 6.3 模型微调

#### 使用预训练权重

```bash
python src/main.py train --data data/wheat_disaster.yaml --weights yolov8n.pt --epochs 100
```

#### 冻结层

```bash
python src/main.py train --data data/wheat_disaster.yaml --weights yolov8n.pt --freeze 10
```

## 7. 常见问题

### 7.1 训练损失不下降

可能原因：
- 学习率过大或过小
- 数据标注错误
- 数据集不平衡
- 模型选择不当

解决方法：
- 调整学习率（lr0）
- 检查和修复数据标注
- 处理不平衡数据集（过采样/欠采样）
- 尝试更大或更小的模型

### 7.2 过拟合

可能原因：
- 数据集太小
- 数据增强不足
- 训练轮数过多

解决方法：
- 增加数据集规模
- 增加数据增强
- 提前停止训练
- 加入正则化

### 7.3 GPU内存不足

可能原因：
- batch size太大
- 输入图像尺寸太大

解决方法：
- 减小batch size
- 减小输入图像尺寸
- 使用更小的模型
- 启用FP16半精度训练

## 8. 训练脚本示例

### 8.1 完整训练脚本

```python
from ultralytics import YOLO
import yaml

def main():
    # 加载配置
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    model = YOLO(config['model']['name'] + '.pt')
    
    # 开始训练
    model.train(
        data='data/wheat_disaster.yaml',
        epochs=config['train']['epochs'],
        batch=config['train']['batch_size'],
        imgsz=config['model']['imgsz'],
        name='wheat_disaster_yolov8',
        save_dir=config['train']['save_dir']
    )
    
    # 验证模型
    model.val(data='data/wheat_disaster.yaml')
    
    # 导出模型
    model.export(format='onnx')

if __name__ == '__main__':
    main()
```

### 8.2 使用命令行训练

```bash
python src/main.py train \
  --data data/wheat_disaster.yaml \
  --weights yolov8n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --name wheat_disaster_yolov8 \
  --save_dir runs/train
```

## 9. 进阶训练技巧

### 9.1 学习率预热

```yaml
train:
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
```

### 9.2 训练策略

- **先冻结后解冻**: 先冻结骨干网络进行微调，再解冻进行全网络训练
- **渐进式训练**: 先使用小尺寸图像训练，逐步切换到较大尺寸
- **知识蒸馏**: 使用大型模型指导小型模型训练

### 9.3 模型压缩

- **剪枝**: 移除模型中不重要的连接
- **量化**: 将模型参数转换为更低的精度
- **蒸馏**: 使用教师模型指导学生模型训练

## 10. 训练后的模型导出

### 10.1 导出为ONNX格式

```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

### 10.2 导出为TensorRT格式

```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format engine
```

### 10.3 导出为CoreML格式

```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format coreml
```

## 11. 模型部署

### 11.1 部署方式

- **服务器部署**: 使用Flask/FastAPI在云端部署模型
- **边缘设备部署**: 在嵌入式设备（如Jetson Nano）上部署
- **移动端部署**: 使用CoreML/TensorFlow Lite在iOS/Android上部署
- **Web部署**: 使用TensorFlow.js在浏览器上部署

### 11.2 性能优化

- **TensorRT**: 使用NVIDIA TensorRT进行推理优化
- **OpenVINO**: 使用Intel OpenVINO进行推理优化
- **Quantization**: 模型量化减少内存使用和提高推理速度

### 11.3 示例部署

```python
# Flask服务部署
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)
model = YOLO('runs/train/exp/weights/best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```