# 配置文件说明

## YAML配置文件

`config/config.yaml`文件包含所有模型和训练相关的配置。

### 模型配置

```yaml
model:
  # 模型名称或路径
  # 可以是：yolov8n, yolov8s, yolov8m, yolov8l, yolov8x 或 .pt文件路径
  name: "yolov8n"
  
  # 输入图像大小
  # 默认值为640，根据硬件性能调整
  imgsz: 640
  
  # 置信度阈值
  # 低于此值的检测结果将被忽略
  # 默认值为0.25
  conf: 0.25
  
  # IOU阈值
  # NMS非极大值抑制中的IOU阈值
  # 默认值为0.45
  iou: 0.45
  
  # 最大检测数
  # 每张图像最多检测的目标数
  # 默认值为300
  max_det: 300
  
  # 设备选择
  # 可以是cpu, cuda, 0（GPU ID）等
  device: "auto"
  
  # 是否启用FP16半精度
  half: false
  
  # 是否使用DNN后端
  dnn: false
```

### 训练配置

```yaml
train:
  # 训练轮数
  # 默认值为100，根据数据集大小调整
  epochs: 100
  
  # 批次大小
  # 根据GPU内存调整
  batch_size: 16
  
  # 初始学习率
  # 默认值为0.01，根据任务调整
  lr0: 0.01
  
  # 学习率衰减
  # 默认值为0.01
  lrf: 0.01
  
  # 动量
  # 梯度下降动量优化器参数
  # 默认值为0.937
  momentum: 0.937
  
  # 权重衰减
  # L2正则化系数，防止过拟合
  weight_decay: 0.0005
  
  # 结果保存目录
  save_dir: "runs/train"
  
  # 每多少轮保存一次权重
  save_period: -1  # 表示仅保存best和last
  
  # 是否保存训练检查点
  save_weights: true
  
  # 是否保存训练后的ONNX模型
  save_onnx: true
  
  # 每多少轮记录一次训练结果
  log_period: 10
  
  # 验证集评估频率
  val_period: 1
  
  # 学习率预热轮数
  warmup_epochs: 3
  
  # 预热阶段动量
  warmup_momentum: 0.8
  
  # 预热阶段偏置学习率
  warmup_bias_lr: 0.1
  
  # 是否启用数据增强
  data_augmentation: true
  
  # 随机翻转概率（左右翻转）
  fliplr: 0.5
  
  # 随机翻转概率（上下翻转）
  flipud: 0.0
  
  # 随机旋转角度
  rotate: 0.0
  
  # 马赛克增强概率
  mosaic: 1.0
  
  # MixUp增强概率
  mixup: 0.0
  
  # 图像缩放范围
  scale: 0.5
  
  # 平移范围
  translate: 0.2
  
  # 剪切范围
  shear: 0.0
  
  # 色域调整（Hue）
  hsv_h: 0.015
  
  # 色域调整（Saturation）
  hsv_s: 0.7
  
  # 色域调整（Value）
  hsv_v: 0.4
```

### 数据集配置

```yaml
dataset:
  # 数据集根目录
  path: "data"
  
  # 训练集图像路径
  train: "train/images"
  
  # 验证集图像路径
  val: "val/images"
  
  # 测试集图像路径
  test: "test/images"
  
  # 类别数量
  nc: 5
  
  # 类别名称
  names: ['rust', 'powdery_mildew', 'aphid', 'wheat_blast', 'healthy']
  
  # 类别颜色（可选）
  # 用于可视化时的边界框颜色
  # colors: [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
  
  # 数据集标签格式
  # 可以是yolo, coco, voc等
  format: "yolo"
  
  # 是否是多类别数据集
  multi_label: false
  
  # 是否启用自动下载
  download: false
```

### 推理配置

```yaml
inference:
  # 是否显示推理结果
  show_results: true
  
  # 结果保存目录
  save_dir: "runs/detect"
  
  # 是否保存检测结果
  save_results: true
  
  # 是否保存带边界框的图像
  save_images: true
  
  # 是否保存txt格式的检测结果
  save_txt: true
  
  # 是否保存CSV格式的检测结果
  save_csv: true
  
  # 边界框线条宽度
  line_width: 2
  
  # 标签字体大小
  font_size: 12
  
  # 置信度阈值
  conf_threshold: 0.25
  
  # IOU阈值
  iou_threshold: 0.45
  
  # 是否启用NMS非极大值抑制
  nms: true
  
  # 是否启用增强推理
  augment: false
  
  # 是否启用多尺度推理
  multi_scale: false
  
  # 是否启用TTA测试时增强
  tta: false
```

### 评估配置

```yaml
eval:
  # 是否显示评估结果
  show: true
  
  # 评估指标
  metrics: ["mAP50", "mAP50-95", "precision", "recall", "f1"]
  
  # 是否绘制混淆矩阵
  plot_confusion_matrix: true
  
  # 是否绘制PR曲线
  plot_pr_curve: true
  
  # 是否绘制R曲线
  plot_r_curve: true
  
  # 评估日志
  log_file: "runs/val/eval.log"
```

### 示例完整配置

```yaml
model:
  name: "yolov8n"
  imgsz: 640
  conf: 0.25
  iou: 0.45
  max_det: 300
  device: "auto"
  half: false
  dnn: false

train:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  save_dir: "runs/train"
  save_period: -1
  save_weights: true
  save_onnx: true
  log_period: 10
  val_period: 1
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  data_augmentation: true
  fliplr: 0.5
  flipud: 0.0
  rotate: 0.0
  mosaic: 1.0
  mixup: 0.0
  scale: 0.5
  translate: 0.2
  shear: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4

dataset:
  path: "data"
  train: "train/images"
  val: "val/images"
  test: "test/images"
  nc: 5
  names: ['rust', 'powdery_mildew', 'aphid', 'wheat_blast', 'healthy']
  format: "yolo"
  multi_label: false
  download: false

inference:
  show_results: true
  save_dir: "runs/detect"
  save_results: true
  save_images: true
  save_txt: true
  save_csv: true
  line_width: 2
  font_size: 12
  conf_threshold: 0.25
  iou_threshold: 0.45
  nms: true
  augment: false
  multi_scale: false
  tta: false

eval:
  show: true
  metrics: ["mAP50", "mAP50-95", "precision", "recall", "f1"]
  plot_confusion_matrix: true
  plot_pr_curve: true
  plot_r_curve: true
  log_file: "runs/val/eval.log"
```

### 参数说明

#### 通用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| imgsz | 输入图像尺寸 | 640 |
| conf | 置信度阈值 | 0.25 |
| iou | IOU阈值 | 0.45 |
| max_det | 最大检测数 | 300 |
| device | 设备选择 | "auto" |

#### 训练参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| epochs | 训练轮数 | 100 |
| batch_size | 批次大小 | 16 |
| lr0 | 初始学习率 | 0.01 |
| lrf | 学习率衰减 | 0.01 |
| momentum | 动量 | 0.937 |
| weight_decay | 权重衰减 | 0.0005 |
| save_dir | 结果保存目录 | "runs/train" |
| save_period | 权重保存周期 | -1 |
| warmup_epochs | 学习率预热 | 3 |
| fliplr | 随机翻转（左右） | 0.5 |
| mosaic | 马赛克增强 | 1.0 |
| mixup | MixUp增强 | 0.0 |

#### 推理参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| show_results | 显示结果 | true |
| save_dir | 结果保存目录 | "runs/detect" |
| save_images | 保存带框图像 | true |
| save_txt | 保存txt结果 | true |
| line_width | 边界框宽度 | 2 |
| augment | 增强推理 | false |

### 使用配置文件

在main.py中加载配置文件：

```python
import yaml

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

config = load_config()
```

### 配置文件示例

```python
from ultralytics import YOLO
import yaml

def main():
    # 加载配置
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    model = YOLO(config['model']['name'] + '.pt')
    
    # 训练
    model.train(
        data='data/wheat_disaster.yaml',
        epochs=config['train']['epochs'],
        batch=config['train']['batch_size'],
        imgsz=config['model']['imgsz'],
        conf=config['model']['conf'],
        iou=config['model']['iou'],
        device=config['model']['device'],
        name='wheat_disaster_yolov8',
        save_dir=config['train']['save_dir']
    )
    
    # 检测
    # results = model.predict(source='examples/test.jpg', conf=config['model']['conf'], iou=config['model']['iou'])
    # results.show()

if __name__ == '__main__':
    main()
```

## 数据集配置文件

`data/wheat_disaster.yaml`文件是YOLOv8训练时使用的数据集配置文件。

```yaml
# 数据集根目录
path: data

train: train/images  # 训练集图像路径
val: val/images      # 验证集图像路径
test: test/images    # 测试集图像路径

# 类别数
nc: 5

# 类别名称
names:
  0: rust              # 小麦锈病
  1: powdery_mildew   # 小麦白粉病
  2: aphid             # 蚜虫
  3: wheat_blast      # 小麦全蚀病
  4: healthy           # 健康小麦

# 可选：数据集描述
# description: "Wheat Disaster Detection Dataset - 5 classes"
# version: "1.0"
# author: "Team Name"
```

### 数据集文件结构

数据集需要按照以下结构存放：

```
data/
├── wheat_disaster.yaml
├── train/
│   ├── images/
│   │   ├── img0001.jpg
│   │   └── img0002.jpg
│   └── labels/
│       ├── img0001.txt
│       └── img0002.txt
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 注意事项

1. **YAML格式**：配置文件需要使用正确的YAML格式，缩进必须严格对齐（使用空格）
2. **相对路径**：所有路径都是相对于项目根目录
3. **类别名称**：类别名称需要与标注文件中的类别ID一一对应
4. **训练配置**：训练配置需要根据硬件性能调整，如batch size
5. **模型选择**：模型大小需要根据硬件和精度需求选择，n/s/m/l/x
6. **学习率**：学习率对训练效果影响很大，需要根据数据集调整
7. **数据增强**：适当的数据增强可以提升模型鲁棒性

## 配置文件常见错误

1. **缩进错误**：YAML文件中缩进必须使用空格，且严格对齐
2. **路径错误**：数据集路径错误或文件不存在
3. **格式错误**：YAML文件格式错误，如使用Tab缩进、缺失冒号等
4. **参数值错误**：参数值类型错误，如将字符串设置给数字类型参数

## 配置文件优化建议

1. **学习率**：根据batch size调整学习率，如batch_size=16时lr0=0.01，batch_size=32时lr0=0.02
2. **batch size**：尽可能使用大batch size以提高训练稳定性
3. **输入尺寸**：对于小目标，可以尝试更大的输入尺寸
4. **数据增强**：根据数据集和任务调整数据增强策略
5. **训练轮数**：根据模型收敛情况调整训练轮数，建议使用提前停止

## 参考资料

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [YOLOv8训练指南](https://docs.ultralytics.com/guides/training/)
- [YOLOv8配置参考](https://docs.ultralytics.com/guides/configuration/)
- [YOLOv8GitHub](https://github.com/ultralytics/ultralytics)