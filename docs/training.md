# 训练指南

## 1. 数据集准备

### 1.1 数据集下载

可以从以下渠道获取小麦灾害数据集：

1. [Kaggle](https://www.kaggle.com/datasets/) - 小麦病虫害数据集
2. [PlantVillage](https://www.plantvillage.org/) - 植物病害数据集
3. [中国农业大学](http://www.cau.edu.cn/) - 小麦灾害数据集
4. 自制数据集 - 自己拍摄小麦图像并标注

### 1.2 数据集标注

可以使用以下工具进行数据集标注：

- [LabelImg](https://github.com/tzutalin/labelImg) - 简单易用的标注工具
- [LabelBox](https://labelbox.com/) - 云标注平台
- [CVAT](https://github.com/opencv/cvat) - 开源视频标注工具

### 1.3 数据集结构

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

### 1.4 数据集比例

建议的数据集比例：

- 训练集: 70%
- 验证集: 20%
- 测试集: 10%

## 2. 训练配置

### 2.1 模型选择

YOLOv8 提供了以下模型：

| 模型名称 | 大小 | 速度 | 精度 |
|---------|------|------|------|
| yolov8n | 最小 | 最快 | 最低 |
| yolov8s | 较小 | 较快 | 较低 |
| yolov8m | 中等 | 中等 | 中等 |
| yolov8l | 较大 | 较慢 | 较高 |
| yolov8x | 最大 | 最慢 | 最高 |

建议：

- 快速原型开发: `yolov8n.pt`
- 部署到移动设备: `yolov8n.pt` 或 `yolov8s.pt`
- 生产环境: `yolov8s.pt` 或 `yolov8m.pt`
- 最高精度: `yolov8l.pt` 或 `yolov8x.pt`

### 2.2 参数配置

#### 2.2.1 训练参数

- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `lr0`: 初始学习率
- `lrf`: 最终学习率
- `momentum`: 动量
- `weight_decay`: 权重衰减
- `warmup_epochs`: 预热轮数

#### 2.2.2 数据增强

数据增强可以提高模型的泛化能力：

- `mixup`: 混合增强
- `mosaic`: 马赛克增强
- `hsv_h`: 色调调整
- `hsv_s`: 饱和度调整
- `hsv_v`: 亮度调整
- `scale`: 缩放增强
- `flipud`: 垂直翻转
- `fliplr`: 水平翻转

#### 2.2.3 损失函数

YOLOv8 使用以下损失函数：

- 分类损失: BCEWithLogitsLoss
- 回归损失: CIoULoss
- 置信度损失: BCEWithLogitsLoss

## 3. 训练命令

### 3.1 基础训练

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100
```

### 3.2 自定义训练

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 200 --batch 32 --lr0 0.01 --imgsz 800 --device 0
```

### 3.3 恢复训练

如果训练中断，可以使用 `--resume` 参数恢复：

```bash
uv run python src/main.py train --resume runs/train/exp/weights/last.pt
```

### 3.4 微调训练

可以在预训练模型的基础上进行微调：

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 50 --weights yolov8m.pt
```

## 4. 训练过程

### 4.1 训练日志

训练过程中会输出以下信息：

- 训练轮数
- 当前批次
- 损失值 (box, cls, dfl)
- 速度 (ms/step, imgs/sec)
- 学习率

### 4.2 训练进度

训练进度可以通过以下方式查看：

- 命令行输出
- TensorBoard 日志
- Weights & Biases 日志

## 5. 训练监控

### 5.1 使用 TensorBoard

```bash
uv run tensorboard --logdir runs/train
```

### 5.2 使用 Weights & Biases

1. 安装 W&B：

```bash
pip install wandb
```

2. 登录 W&B：

```bash
wandb login
```

3. 训练时添加 `--wandb` 参数：

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100 --wandb
```

## 6. 训练结果

### 6.1 训练结果目录

训练结果会保存在 `runs/train/exp` 目录下：

```
exp/
├── weights/            # 模型权重
│   ├── best.pt         # 最佳模型
│   └── last.pt         # 最后模型
├── results.csv         # 训练结果日志
├── confusion_matrix.png # 混淆矩阵
├── labels_correlogram.jpg # 标签相关性
└── train_batch0.jpg    # 训练样本示例
```

### 6.2 评估指标

训练完成后会输出评估指标：

- mAP@0.5: 0.5 IoU 阈值下的平均精度
- mAP@0.5:0.95: 0.5 到 0.95 IoU 阈值下的平均精度
- F1: F1 得分
- Precision: 精确率
- Recall: 召回率

## 7. 调优策略

### 7.1 学习率调优

学习率对模型训练非常重要。如果学习率过高，模型可能无法收敛；如果学习率过低，模型训练会非常慢。

可以使用以下命令进行学习率分析：

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --lr0 0.01 --lrf 0.001 --epochs 50
```

### 7.2 数据增强调优

数据增强可以提高模型的泛化能力。可以根据以下原则调整数据增强：

- 增加旋转、缩放和翻转增强
- 调整颜色空间
- 增加噪声

### 7.3 模型微调

如果模型效果不理想，可以尝试以下方法：

1. 增加训练轮数
2. 调整学习率
3. 增加更多的数据
4. 使用更复杂的模型

## 8. 常见问题

### 8.1 模型不收敛

原因：

- 学习率过高
- 数据集过小
- 数据增强过度
- 模型太复杂

解决方法：

1. 降低学习率
2. 增加数据量
3. 减少数据增强
4. 使用更简单的模型

### 8.2 过拟合

原因：

- 数据集太小
- 训练轮数过多
- 模型太复杂

解决方法：

1. 增加数据量
2. 提前停止训练
3. 使用数据增强
4. 使用正则化

### 8.3 显存不足

原因：

- 批次大小过大
- 输入图像尺寸过大
- 模型太复杂

解决方法：

1. 减小批次大小
2. 减小输入图像尺寸
3. 使用更小的模型
4. 使用半精度训练

## 9. 训练脚本示例

以下是一个完整的训练脚本示例：

```python
from ultralytics import YOLO
import yaml

# 加载配置
with open('./src/config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载预训练模型
model = YOLO('yolov8m.pt')

# 训练模型
results = model.train(
    data=config['data']['path'],
    epochs=config['training']['epochs'],
    batch=config['training']['batch_size'],
    imgsz=config['model']['imgsz'],
    lr0=config['training']['lr0'],
    project=config['training']['save_dir'],
    name='exp',
    exist_ok=True
)

# 验证模型
metrics = model.val()

# 保存模型
model.export(format='onnx')
```

## 10. 进阶训练技巧

### 10.1 半精度训练

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --half
```

### 10.2 多尺度训练

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --multi-scale
```

### 10.3 自定义训练循环

可以通过自定义训练循环对训练过程进行更细粒度的控制：

```python
model = YOLO('yolov8m.pt')
for epoch in range(100):
    model.train(epochs=1, resume=True, ...)
```

## 11. 模型导出

### 11.1 导出 ONNX

```bash
uv run python src/main.py export --weights best.pt --format onnx
```

### 11.2 导出 TensorRT

```bash
uv run python src/main.py export --weights best.pt --format engine
```

### 11.3 导出 CoreML

```bash
uv run python src/main.py export --weights best.pt --format coreml
```

## 12. 部署

可以将模型部署到以下平台：

- 服务器端: Python, ONNX, TensorRT
- 移动端: CoreML, TensorFlow Lite
- 嵌入式系统: OpenVINO, TensorRT
- Web 端: ONNX.js, TensorFlow.js

## 13. 参考资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv8 训练教程](https://docs.ultralytics.com/modes/train/)
- [YOLOv8 数据格式](https://docs.ultralytics.com/datasets/detect/)
- [YOLOv8 模型选择](https://docs.ultralytics.com/models/yolov8/)

## 14. 常见问题

### 14.1 如何提高模型精度

1. 增加训练轮数
2. 调整学习率
3. 使用更大的模型
4. 增加数据集
5. 增加数据增强

### 14.2 如何提高模型速度

1. 使用更小的模型
2. 减小输入图像尺寸
3. 使用半精度或 INT8 量化
4. 使用 TensorRT 加速

### 14.3 如何避免过拟合

1. 增加数据量
2. 使用数据增强
3. 使用正则化
4. 提前停止训练

### 14.4 如何处理类别不平衡

1. 调整损失函数
2. 使用 Focal Loss
3. 增加少数类别的数据

### 14.5 如何处理低对比度图像

1. 调整直方图均衡化
2. 使用 CLAHE 算法
3. 调整图像饱和度

### 14.6 如何处理光照变化

1. 调整 HSV 颜色空间
2. 使用自适应阈值
3. 使用多尺度融合

### 14.7 如何处理模糊图像

1. 使用图像增强
2. 使用去模糊算法
3. 使用 CNN 进行图像恢复

## 15. 下一步

- 阅读 [快速开始](quick-start.md) 了解基本使用方法
- 阅读 [API 文档](api.md) 了解更多接口
- 阅读 [配置说明](configuration.md) 了解更多配置选项
- 阅读 [数据集结构](dataset-structure.md) 了解更多数据集信息
- 阅读 [常见问题](faq.md) 获取更多解决方案

## 16. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)