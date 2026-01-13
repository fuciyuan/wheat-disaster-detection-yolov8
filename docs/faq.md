# 常见问题

## 1. 环境配置

### 1.1 如何安装 uv？

在 Windows 系统上，可以通过以下命令安装 uv：

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

在 Linux/macOS 系统上，可以通过以下命令安装 uv：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 如何安装项目依赖？

```bash
uv sync
```

### 1.3 如何创建虚拟环境？

```bash
uv venv
```

### 1.4 如何激活虚拟环境？

在 Windows 系统上，激活虚拟环境：

```bash
uv venv
.venv\Scripts\activate
```

在 Linux/macOS 系统上，激活虚拟环境：

```bash
uv venv
source .venv/bin/activate
```

### 1.5 如何更新项目依赖？

```bash
uv add ultralytics==8.2.0
```

### 1.6 如何安装特定版本的 PyTorch？

```bash
uv add torch==2.3.0
```

### 1.7 如何安装支持 CUDA 的 PyTorch？

```bash
uv add torch==2.3.0+cu118 -i https://download.pytorch.org/whl/cu118
uv add torchvision==0.18.0+cu118 -i https://download.pytorch.org/whl/cu118
```

## 2. 模型训练

### 2.1 如何开始训练模型？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100
```

### 2.2 如何恢复训练？

```bash
uv run python src/main.py train --resume runs/train/exp/weights/last.pt
```

### 2.3 如何进行模型微调？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 50 --weights yolov8m.pt
```

### 2.4 如何调整批次大小？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100 --batch 32
```

### 2.5 如何调整学习率？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100 --lr0 0.001
```

### 2.6 如何训练更大的模型？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 200 --weights yolov8m.pt
```

### 2.7 如何使用 TensorBoard？

```bash
uv run tensorboard --logdir runs/train
```

### 2.8 如何使用 Weights & Biases？

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

### 2.9 如何停止训练？

在命令行中按下 `Ctrl+C` 可以停止训练。

## 3. 模型检测

### 3.1 如何进行单张图像检测？

```bash
uv run python src/main.py detect --image test.jpg
```

### 3.2 如何进行批量检测？

```bash
uv run python src/main.py detect --images ./data/test/images
```

### 3.3 如何进行视频检测？

```bash
uv run python src/main.py detect --video test.mp4
```

### 3.4 如何进行实时摄像头检测？

```bash
uv run python src/main.py detect --camera
```

### 3.5 如何调整检测阈值？

```bash
uv run python src/main.py detect --image test.jpg --conf 0.5
```

### 3.6 如何保存检测结果？

```bash
uv run python src/main.py detect --image test.jpg --save
```

### 3.7 如何修改模型路径？

```bash
uv run python src/main.py detect --image test.jpg --model runs/train/exp/weights/best.pt
```

### 3.8 如何使用半精度推理？

```bash
uv run python src/main.py detect --image test.jpg --half
```

### 3.9 如何使用 GPU 加速？

确保安装了 CUDA 和 PyTorch，可以通过以下命令检查：

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 True，则表示支持 GPU 加速。

## 4. 数据集

### 4.1 如何准备数据集？

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

### 4.2 如何标注数据集？

可以使用以下工具进行标注：

- [LabelImg](https://github.com/tzutalin/labelImg)
- [LabelBox](https://labelbox.com/)
- [CVAT](https://github.com/opencv/cvat)

### 4.3 数据集标签格式是什么？

每个标签文件包含如下格式：

```
class_id x_center y_center width height
```

例如：
```
0 0.1 0.2 0.3 0.4
1 0.5 0.6 0.7 0.8
```

### 4.4 如何划分训练集和验证集？

建议的划分比例：

- 训练集: 70%
- 验证集: 20%
- 测试集: 10%

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

### 4.5 如何解决类别不平衡问题？

类别不平衡会影响模型性能，可以通过以下方式解决：

1. 调整损失函数
2. 使用 Focal Loss
3. 增加少数类别的数据
4. 使用数据增强

### 4.6 如何提高数据集质量？

可以通过以下方式提高数据集质量：

1. 删除模糊或损坏的图像
2. 修正错误的标签
3. 增加更多的样本
4. 增加多样性

## 5. 性能优化

### 5.1 如何提高推理速度？

可以通过以下方式提高推理速度：

1. 使用更小的模型
2. 减小输入图像尺寸
3. 使用半精度或 INT8 量化
4. 使用 TensorRT 加速

### 5.2 如何提高模型精度？

可以通过以下方式提高模型精度：

1. 增加训练轮数
2. 调整学习率
3. 使用更大的模型
4. 增加更多的数据
5. 增加数据增强

### 5.3 如何避免过拟合？

可以通过以下方式避免过拟合：

1. 增加数据量
2. 使用数据增强
3. 使用正则化
4. 提前停止训练

### 5.4 如何提高训练速度？

可以通过以下方式提高训练速度：

1. 使用 GPU 加速
2. 增加批次大小
3. 使用更大的 batch
4. 使用更快的数据加载器

### 5.5 如何使用 TensorRT 加速模型？

1. 导出模型为 ONNX 格式：

```bash
uv run python src/main.py export --weights best.pt --format onnx
```

2. 使用 TensorRT 进行优化：

```python
import tensorrt as trt

# 加载 ONNX 模型
builder = trt.Builder(trt.Logger())
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, builder.logger)
with open('best.onnx', 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# 构建引擎
build_config = builder.create_builder_config()
build_config.max_workspace_size = 1 << 30
engine = builder.build_engine(network, build_config)
```

### 5.6 如何使用半精度训练？

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --half
```

### 5.7 如何使用 INT8 量化？

```bash
uv run python src/main.py export --weights best.pt --format engine --int8
```

### 5.8 如何使用模型蒸馏？

可以通过模型蒸馏提高模型性能：

```python
from ultralytics import YOLO
import torch

# 加载教师模型
teacher_model = YOLO('yolov8x.pt')
# 加载学生模型
student_model = YOLO('yolov8n.pt')

# 模型蒸馏代码
def distill(teacher_model, student_model, data, epochs):
    for epoch in range(epochs):
        # 训练学生模型
        student_model.train(data=data, epochs=1, teacher_model=teacher_model)

distill(teacher_model, student_model, 'src/data/wheat_disaster.yaml', 100)
```

## 6. 模型导出

### 6.1 如何导出为 ONNX？

```bash
uv run python src/main.py export --weights best.pt --format onnx
```

### 6.2 如何导出为 TensorRT？

```bash
uv run python src/main.py export --weights best.pt --format engine
```

### 6.3 如何导出为 CoreML？

```bash
uv run python src/main.py export --weights best.pt --format coreml
```

### 6.4 如何导出为 TensorFlow Lite？

```bash
uv run python src/main.py export --weights best.pt --format tflite
```

### 6.5 如何导出为 Pytorch？

```bash
uv run python src/main.py export --weights best.pt --format torchscript
```

### 6.6 如何导出为 ONNX 并简化？

```bash
uv run python src/main.py export --weights best.pt --format onnx --simplify
```

### 6.7 如何导出为动态批量 ONNX？

```bash
uv run python src/main.py export --weights best.pt --format onnx --dynamic
```

### 6.8 如何导出为 TensorRT 并优化？

```bash
uv run python src/main.py export --weights best.pt --format engine --workspace 8
```

### 6.9 如何导出为 PyTorch 并量化？

```bash
uv run python src/main.py export --weights best.pt --format torchscript --int8
```

### 6.10 如何使用导出的模型？

```python
from ultralytics import YOLO

model = YOLO('best.onnx')
results = model('test.jpg')
results.save()
```

## 7. 常见错误

### 7.1 FileNotFoundError

```
FileNotFoundError: [Errno 2] No such file or directory: 'test.jpg'
```

解决方法：

- 检查文件路径是否正确
- 确保文件存在

### 7.2 ImportError

```
ImportError: No module named 'ultralytics'
```

解决方法：

- 安装 ultralytics

```bash
uv add ultralytics==8.2.0
```

### 7.3 RuntimeError

```
RuntimeError: CUDA out of memory.
```

解决方法：

- 减小批次大小
- 减小输入图像尺寸
- 使用更小的模型
- 使用半精度训练

### 7.4 TypeError

```
TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'
```

解决方法：

- 检查变量是否为 None
- 确保输入参数正确

### 7.5 KeyError

```
KeyError: 'model'
```

解决方法：

- 检查配置文件
- 确保包含 'model' 键

### 7.6 AttributeError

```
AttributeError: 'NoneType' object has no attribute 'detect'
```

解决方法：

- 确保模型已成功加载
- 检查模型路径是否正确

### 7.7 ValueError

```
ValueError: unsupported value: [None, None, None, None, None]
```

解决方法：

- 检查输入图像是否为空
- 确保输入图像路径正确

### 7.8 MemoryError

```
MemoryError: could not allocate memory
```

解决方法：

- 减小批次大小
- 减小输入图像尺寸
- 使用更小的模型

### 7.9 OSError

```
OSError: [WinError 123] 文件名、目录名或卷标语法不正确
```

解决方法：

- 检查文件路径是否正确
- 确保不包含特殊字符

### 7.10 TypeError

```
TypeError: detect() missing 1 required positional argument: 'image_path'
```

解决方法：

- 检查函数参数
- 确保所有参数正确

## 8. 部署相关

### 8.1 如何部署到服务器？

可以通过以下方式部署模型：

1. 使用 Flask
2. 使用 FastAPI
3. 使用 Django

以下是 Flask 部署示例：

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

### 8.2 如何部署到移动设备？

可以通过以下方式部署到移动设备：

1. 使用 CoreML
2. 使用 TensorFlow Lite

### 8.3 如何部署到边缘设备？

可以通过以下方式部署到边缘设备：

1. 使用 TensorRT
2. 使用 OpenVINO
3. 使用 TensorFlow Lite

### 8.4 如何使用 Docker 部署？

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 5000

CMD ["uv", "run", "python", "app.py"]
```

### 8.5 如何使用 Kubernetes 部署？

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wheat-detection
  labels:
    app: wheat-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: wheat-detection
  template:
    metadata:
      labels:
        app: wheat-detection
    spec:
      containers:
      - name: wheat-detection
        image: wheat-detection:latest
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: wheat-detection-service
spec:
  type: LoadBalancer
  selector:
    app: wheat-detection
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
```

## 9. 下一步

- 阅读 [快速开始](quick-start.md) 了解基本使用方法
- 阅读 [训练指南](training.md) 了解更多训练技巧
- 阅读 [API 文档](api.md) 了解更多接口
- 阅读 [配置说明](configuration.md) 了解更多配置选项
- 阅读 [数据集结构](dataset-structure.md) 了解更多数据集信息

## 10. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)