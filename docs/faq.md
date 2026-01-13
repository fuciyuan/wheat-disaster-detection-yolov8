# 常见问题

## 1. 环境配置

### Q: 如何安装uv包管理器？

A: 在Windows上，可以通过以下命令安装uv：
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Q: 运行uv sync时出现网络问题怎么办？

A: 可以尝试使用国内镜像：
```bash
uv pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
uv sync
```

### Q: 如何验证安装是否成功？

A: 运行测试脚本：
```bash
python test.py
```

如果安装成功，会显示：
```
小麦灾害检测系统初始化成功！
```

### Q: 如何检查Python版本？

A: 使用命令：
```bash
python --version
```

推荐使用Python 3.10或更高版本。

## 2. 模型训练

### Q: 训练过程中遇到`CUDA out of memory`怎么办？

A: 解决方法：
1. 减小batch size：从16减小到8或4
2. 减小输入图像尺寸：从640减小到416或320
3. 使用更小的模型：从yolov8s切换到yolov8n
4. 启用FP16：在训练命令中添加 `--half` 参数

### Q: 训练损失不下降怎么办？

A: 可能原因及解决方法：
- 学习率不合适：尝试减小或增大学习率（lr0）
- 数据质量问题：检查数据标注是否正确
- 数据集过小：增加训练样本
- 模型过大：尝试使用更小的模型

### Q: 模型过拟合怎么办？

A: 解决方法：
1. 增加数据增强：调整mosaic、mixup等参数
2. 减小模型规模：使用更小的模型
3. 提前停止训练：在验证集性能下降时停止
4. 添加正则化：增大weight_decay值

### Q: 训练集和验证集准确率差异大怎么办？

A: 这可能是过拟合的表现。解决方法：
- 增加数据增强
- 减小模型复杂度
- 使用Dropout层
- 提前停止训练

### Q: 如何在CPU上训练模型？

A: 使用 `--device cpu` 参数：
```bash
python src/main.py train --data data/wheat_disaster.yaml --epochs 100 --device cpu
```

## 3. 模型检测

### Q: 运行检测时没有显示结果？

A: 检查步骤：
1. 确认输入图像路径正确
2. 检查置信度阈值是否设置过低
3. 确认模型文件存在
4. 检查是否有权限读取图像文件

### Q: 检测结果边界框不准确？

A: 解决方法：
1. 增加训练轮数
2. 调整置信度阈值
3. 优化数据标注
4. 使用更大的模型

### Q: 如何批量检测多张图像？

A: 使用 `--images` 参数：
```bash
python src/main.py detect --images data/test/images
```

### Q: 如何保存检测结果？

A: 使用 `--save` 参数：
```bash
python src/main.py detect --image test.jpg --save
```

## 4. 数据集相关

### Q: 数据集如何组织？

A: 数据集需要按照以下结构：
```
data/
├── wheat_disaster.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Q: 标签文件格式如何？

A: 使用YOLO格式：
- 每个目标一行
- 格式：`class_id x_center y_center width height`
- 坐标已归一化到0-1之间

### Q: 如何将COCO格式转换为YOLO格式？

A: 使用ultralytics提供的工具：
```python
from ultralytics.data.converter import convert_coco
convert_coco('path/to/coco/annotations', 'path/to/output')
```

### Q: 如何处理数据集不平衡？

A: 解决方法：
1. 数据采样：过采样或欠采样
2. 类别加权：训练时设置不同的类别权重
3. 数据增强：为少数类别添加更多增强

## 5. 性能优化

### Q: 如何提高模型检测速度？

A: 优化方法：
1. 使用更小的模型：yolov8n
2. 减小输入图像尺寸：416或320
3. 启用FP16半精度推理
4. 量化模型到INT8
5. 使用TensorRT加速

### Q: 如何提高模型精度？

A: 优化方法：
1. 使用更大的模型：yolov8s, yolov8m
2. 增大输入图像尺寸：800或1024
3. 增加训练轮数
4. 优化数据集
5. 使用数据增强

### Q: 如何加速推理？

A: 加速方法：
- ONNX Runtime
- TensorRT
- OpenVINO
- CoreML
- TensorFlow Lite

## 6. 模型导出

### Q: 如何导出模型为ONNX格式？

A: 使用export命令：
```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

### Q: 如何导出模型为TensorRT格式？

A: 使用export命令：
```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format engine
```

### Q: 如何在C++中使用模型？

A: 需要将模型导出为ONNX或TensorRT格式，然后在C++代码中使用相应的推理引擎。

### Q: 如何在移动设备上部署？

A: 导出为CoreML或TensorFlow Lite格式：
```bash
python src/main.py export --weights runs/train/exp/weights/best.pt --format coreml
```

## 7. 常见错误

### Q: 运行时出现 `AttributeError: 'list' object has no attribute 'boxes'`？

A: 这通常是因为模型返回了多个结果。需要遍历结果列表：

```python
results = model(img)
for result in results:
    if result.boxes:
        # 处理检测结果
```

### Q: 运行时出现 `ModuleNotFoundError: No module named 'ultralytics'`？

A: 需要安装ultralytics：
```bash
uv pip install ultralytics
```

### Q: 运行时出现 `FileNotFoundError: models/yolov8n.pt`？

A: 解决方法：
1. 确保模型文件存在
2. 使用自动下载：
```python
from ultralytics import YOLO
YOLO('yolov8n.pt')
```

### Q: 运行时出现 `RuntimeError: CUDA error: out of memory`？

A: 解决方法：
1. 减小batch size
2. 减小输入尺寸
3. 使用更小的模型

### Q: 运行时出现 `ValueError: Expected 1 or more classes in dataset`？

A: 检查数据集中是否有标签文件，以及yaml配置文件是否正确。

### Q: 运行时出现 `AttributeError: 'str' object has no attribute 'shape'`？

A: 可能是传递了字符串路径给模型，而不是图像数组。需要先读取图像：

```python
import cv2
img = cv2.imread('test.jpg')
results = model(img)
```

## 8. 部署相关

### Q: 如何在Flask中部署模型？

A: 示例Flask服务：
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

### Q: 如何在Docker中部署？

A: 创建Dockerfile：
```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]
```

### Q: 如何在Jetson Nano上部署？

A: 需要安装JetPack SDK，然后按照以下步骤：
1. 安装ultralytics
2. 转换模型为TensorRT格式
3. 使用TensorRT进行推理

## 9. 其他问题

### Q: YOLOv8支持哪些Python版本？

A: YOLOv8支持Python 3.8及以上版本。

### Q: 如何更新ultralytics库？

A: 使用以下命令：
```bash
uv pip install -U ultralytics
```

### Q: 如何在Colab中使用YOLOv8？

A: 在Colab中执行以下命令：
```bash
!pip install ultralytics
!git clone https://github.com/ultralytics/ultralytics.git
```

### Q: 如何处理视频流？

A: 使用VideoCapture类处理视频：
```python
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    cv2.imshow('frame', results[0].plot())
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### Q: 如何在摄像头实时检测？

A: 使用命令：
```bash
python src/main.py detect --camera
```

## 10. 性能比较

### Q: YOLOv8各模型大小比较

| 模型 | 参数 | 速度 | 精度 | 适用场景 |
| --- | --- | --- | --- | --- |
| yolov8n | 3.2M | 最快 | 低 | 边缘设备，实时应用 |
| yolov8s | 11.2M | 快 | 较高 | 大多数场景 |
| yolov8m | 25.9M | 中等 | 高 | 精度优先 |
| yolov8l | 43.7M | 慢 | 很高 | 高要求场景 |
| yolov8x | 68.2M | 最慢 | 最高 | 极致精度 |

## 11. 常见数据集格式

### Q: COCO格式

A: COCO格式使用JSON文件存储标注，每个目标包含：
- 类别ID
- 边界框
- 分割点

### Q: Pascal VOC格式

A: Pascal VOC格式使用XML文件存储标注，每个文件对应一张图像。

### Q: YOLO格式

A: YOLO格式使用TXT文件存储标注，每个目标一行。

## 12. 参考资料

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLOv8教程](https://docs.ultralytics.com/guides/)
- [YOLOv8论文](https://arxiv.org/abs/2305.07926)

## 13. 更多帮助

如果您的问题未在此文档中找到解答，可以通过以下方式获取帮助：

1. **GitHub Issues**：在项目仓库提交Issue
2. **社区论坛**：访问ultralytics的Discord和GitHub讨论区
3. **官方文档**：[ultralytics官方文档](https://docs.ultralytics.com/)
4. **教程**：查看GitHub上的相关教程和示例
