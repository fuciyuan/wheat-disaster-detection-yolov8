# 快速开始

## 1. 项目初始化

### 1.1 克隆项目

```bash
git clone https://github.com/syster-0/wheat-disaster-detection-yolov8.git
cd wheat-disaster-detection-yolov8
```

### 1.2 安装 uv 包管理器

在 Windows 系统上，可以通过以下命令安装 uv：

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

在 Linux/macOS 系统上，可以通过以下命令安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.3 安装项目依赖

```bash
uv sync
```

### 1.4 验证安装

```bash
uv run python test.py
```

如果安装成功，会看到以下输出：

```
小麦灾害检测系统初始化成功！

使用示例:
  检测单张图像: python src/main.py detect --image test.jpg
  训练模型: python src/main.py train --data data/wheat_disaster.yaml --epochs 100
```

## 2. 数据集准备

### 2.1 数据集结构

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

### 2.2 数据集配置文件

在 `src/data/wheat_disaster.yaml` 文件中配置数据集：

```yaml
path: ./data/wheat_disaster_dataset
train: images/train
val: images/val
test: images/test
nc: 5
names: [
  "wheat_rust", 
  "powdery_mildew", 
  "leaf_spot", 
  "aphid_damage", 
  "healthy"
]
```

### 2.3 数据集格式

每个标签文件包含如下格式：

```
class_id x_center y_center width height
```

例如：
```
0 0.1 0.2 0.3 0.4
1 0.5 0.6 0.7 0.8
```

## 3. 模型训练

### 3.1 下载预训练模型

```bash
uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3.2 运行训练

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100
```

### 3.3 训练参数

可以通过以下参数调整训练过程：

- `--epochs`: 训练轮数，默认 100
- `--batch`: 批次大小，默认 16
- `--lr0`: 初始学习率，默认 0.01
- `--imgsz`: 输入图像尺寸，默认 640
- `--device`: 使用的设备，默认自动检测
- `--save_dir`: 训练结果保存目录，默认 `runs/train`

例如：

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 200 --batch 32 --imgsz 800
```

## 4. 模型检测

### 4.1 单张图像检测

```bash
uv run python src/main.py detect --image test.jpg
```

### 4.2 批量图像检测

```bash
uv run python src/main.py detect --images ./data/test/images
```

### 4.3 视频检测

```bash
uv run python src/main.py detect --video test.mp4
```

### 4.4 实时摄像头检测

```bash
uv run python src/main.py detect --camera
```

## 5. 模型评估

### 5.1 评估模型

```bash
uv run python src/main.py val --data src/data/wheat_disaster.yaml
```

### 5.2 评估结果

评估结果包含以下指标：

- mAP: 平均精度
- F1: F1 得分
- Precision: 精确率
- Recall: 召回率

## 6. 常见问题

### 6.1 环境配置问题

如果遇到环境配置问题，可以尝试以下解决方案：

1. 确保 Python 版本 >= 3.10
2. 确保所有依赖都已正确安装
3. 运行 `test.py` 检查是否有错误

### 6.2 模型下载失败

如果无法下载预训练模型，可以手动从 [ultralytics](https://github.com/ultralytics/assets/releases/download/v0.0.0) 下载。

### 6.3 检测结果不准确

如果模型的检测结果不准确，可以尝试以下优化方案：

1. 增加训练轮数
2. 调整学习率
3. 使用更大的模型 (例如: `yolov8s.pt`, `yolov8m.pt`)
4. 增加数据集
5. 增加数据增强

### 6.4 显存不足

如果训练过程中显存不足，可以尝试以下优化方案：

1. 减小批次大小
2. 减小输入图像尺寸
3. 使用更小的模型 (例如: `yolov8n.pt`)
4. 使用 `--half` 参数启用半精度训练

## 7. 性能优化

### 7.1 使用 GPU 加速

确保安装了 PyTorch 并且支持 GPU，可以通过以下命令检查：

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 7.2 模型加速

可以通过以下方式加速模型：

1. 使用 ONNX 格式导出模型
2. 使用 TensorRT 加速模型
3. 使用 OpenVINO 优化模型

## 8. 部署

### 8.1 导出模型

```bash
uv run python src/main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

### 8.2 部署到服务器

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 8.3 部署到移动设备

可以将模型导出为 CoreML 或 TensorFlow Lite 格式，然后部署到移动设备。

```bash
uv run python src/main.py export --weights best.pt --format coreml
```

## 9. 下一步

- 阅读 [训练指南](training.md) 获取更多训练技巧
- 阅读 [API 文档](api.md) 了解更多接口
- 阅读 [配置说明](configuration.md) 了解更多配置选项
- 阅读 [数据集结构](dataset-structure.md) 了解更多数据集信息
- 阅读 [常见问题](faq.md) 获取更多解决方案

## 10. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)