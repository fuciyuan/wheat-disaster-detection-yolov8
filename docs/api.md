# API 文档

## 1. 主程序接口

### 1.1 main() 函数

`main()` 函数是程序的入口点，用于解析命令行参数并执行相应的操作。

```python
def main() -> None:
    """
    主程序入口点
    
    解析命令行参数并执行相应的操作
    """
    pass
```

### 1.2 load_model() 函数

`load_model()` 函数用于加载 YOLOv8 模型。

```python
def load_model(model_name: str = 'yolov8n.pt') -> YOLO:
    """
    加载 YOLOv8 模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        YOLO: 加载完成的模型
    """
    pass
```

### 1.3 detect_image() 函数

`detect_image()` 函数用于检测图像中的小麦灾害。

```python
def detect_image(model: YOLO, image_path: str) -> Results:
    """
    检测图像中的小麦灾害
    
    Args:
        model: 模型
        image_path: 图像路径
        
    Returns:
        Results: 检测结果
    """
    pass
```

## 2. 命令行参数

### 2.1 主命令

```bash
uv run python src/main.py --help
```

### 2.2 detect 命令

```bash
uv run python src/main.py detect --image test.jpg
```

- `--image`: 图像路径
- `--images`: 图像目录
- `--video`: 视频路径
- `--camera`: 使用摄像头

### 2.3 train 命令

```bash
uv run python src/main.py train --data data/wheat_disaster.yaml --epochs 100
```

- `--data`: 数据集配置文件
- `--epochs`: 训练轮数
- `--batch`: 批次大小
- `--lr0`: 初始学习率
- `--imgsz`: 输入图像尺寸
- `--device`: 使用的设备
- `--save_dir`: 训练结果保存目录
- `--weights`: 预训练模型路径
- `--resume`: 恢复训练

### 2.4 val 命令

```bash
uv run python src/main.py val --data data/wheat_disaster.yaml
```

- `--data`: 数据集配置文件
- `--weights`: 模型路径
- `--batch`: 批次大小
- `--imgsz`: 输入图像尺寸
- `--device`: 使用的设备

### 2.5 export 命令

```bash
uv run python src/main.py export --weights best.pt --format onnx
```

- `--weights`: 模型路径
- `--format`: 导出格式
- `--device`: 使用的设备
- `--imgsz`: 输入图像尺寸

## 3. 数据结构

### 3.1 检测结果

```python
class Results:
    """
    检测结果
    
    Attributes:
        path: 图像路径
        names: 类别名称
        boxes: 边界框
        conf: 置信度
        cls: 类别索引
    """
    pass
```

### 3.2 边界框

```python
class Boxes:
    """
    边界框
    
    Attributes:
        xyxy: 边界框坐标 (x1, y1, x2, y2)
        xywh: 边界框坐标 (x_center, y_center, width, height)
        conf: 置信度
        cls: 类别索引
    """
    pass
```

## 4. 配置文件

### 4.1 config.yaml

```yaml
model:
  name: 'yolov8n.pt'
  imgsz: 640
  conf: 0.25
  iou: 0.45

training:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  lrf: 0.001
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  save_dir: ./runs/train

data:
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

### 4.2 wheat_disaster.yaml

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

## 5. 模型 API

### 5.1 WheatModel 类

```python
class WheatModel:
    """
    小麦灾害检测模型
    
    Attributes:
        model: YOLO 模型
        conf: 置信度阈值
        iou: IOU 阈值
        names: 类别名称
    """
    def __init__(self, model_name: str = 'yolov8n.pt') -> None:
        """
        初始化模型
        
        Args:
            model_name: 模型名称
        """
        pass
    
    def detect(self, image: np.ndarray) -> Results:
        """
        检测图像中的小麦灾害
        
        Args:
            image: 输入图像
            
        Returns:
            Results: 检测结果
        """
        pass
    
    def train(self, data: str, epochs: int = 100) -> TrainingResults:
        """
        训练模型
        
        Args:
            data: 数据集配置文件
            epochs: 训练轮数
            
        Returns:
            TrainingResults: 训练结果
        """
        pass
    
    def val(self, data: str) -> ValidationResults:
        """
        验证模型
        
        Args:
            data: 数据集配置文件
            
        Returns:
            ValidationResults: 验证结果
        """
        pass
    
    def export(self, format: str = 'onnx') -> ExportResults:
        """
        导出模型
        
        Args:
            format: 导出格式
            
        Returns:
            ExportResults: 导出结果
        """
        pass
```

### 5.2 YOLO 类

YOLO 类是 ultralytics 库提供的核心类，包含以下主要方法：

1. **model.train()**: 训练模型
2. **model.val()**: 验证模型
3. **model.predict()**: 进行推理
4. **model.export()**: 导出模型
5. **model.info()**: 查看模型信息

## 6. 错误处理

### 6.1 常见错误

1. **FileNotFoundError**: 图像文件不存在
2. **ValueError**: 输入参数错误
3. **IOError**: 无法读取图像
4. **ImportError**: 缺少依赖库
5. **RuntimeError**: 运行时错误

### 6.2 异常捕获

```python
def main() -> None:
    try:
        # 检测图像
        result = detect_image(model, args.image)
    except FileNotFoundError:
        print(f"图像文件 {args.image} 不存在。")
    except ValueError:
        print(f"输入参数错误。")
    except IOError:
        print(f"无法读取图像。")
    except ImportError:
        print(f"缺少依赖库。")
    except RuntimeError:
        print(f"运行时错误。")
```

## 7. 性能指标

### 7.1 评估指标

1. **mAP@0.5**: 0.5 IoU 阈值下的平均精度
2. **mAP@0.5:0.95**: 0.5 到 0.95 IoU 阈值下的平均精度
3. **F1**: F1 得分
4. **Precision**: 精确率
5. **Recall**: 召回率

### 7.2 性能指标示例

```python
from ultralytics import YOLO

model = YOLO('best.pt')
metrics = model.val()

print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"F1: {metrics.box.f1:.3f}")
print(f"Precision: {metrics.box.p:.3f}")
print(f"Recall: {metrics.box.r:.3f}")
```

### 7.3 性能比较

| 模型名称 | mAP@0.5 | mAP@0.5:0.95 | F1 | Precision | Recall |
|---------|---------|--------------|----|-----------|--------|
| yolov8n | 0.821 | 0.577 | 0.747 | 0.705 | 0.800 |
| yolov8s | 0.855 | 0.623 | 0.776 | 0.745 | 0.821 |
| yolov8m | 0.876 | 0.665 | 0.804 | 0.779 | 0.838 |
| yolov8l | 0.887 | 0.690 | 0.816 | 0.793 | 0.847 |
| yolov8x | 0.893 | 0.700 | 0.823 | 0.799 | 0.853 |

## 8. 部署 API

### 8.1 Flask 部署

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    """
    检测图像中的小麦灾害
    
    Request:
        image: 上传的图像
        
    Response:
        results: 检测结果
    """
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 8.2 FastAPI 部署

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()
model = YOLO('best.pt')

@app.post('/detect')
def detect(file: UploadFile = File(...)):
    """
    检测图像中的小麦灾害
    
    Args:
        file: 上传的图像
        
    Returns:
        results: 检测结果
    """
    img = Image.open(io.BytesIO(file.file.read()))
    results = model(img)
    return results

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
```

### 8.3 部署脚本

```bash
uv run python -m flask run --host 0.0.0.0 --port 5000
```

## 9. 客户端 API

### 9.1 Python 客户端

```python
import requests

url = 'http://localhost:5000/detect'
files = {'image': open('test.jpg', 'rb')}
response = requests.post(url, files=files)
results = response.json()
print(results)
```

### 9.2 JavaScript 客户端

```javascript
fetch('http://localhost:5000/detect', {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => {
    console.log(data);
  });
```

## 10. 下一步

- 阅读 [快速开始](quick-start.md) 了解基本使用方法
- 阅读 [训练指南](training.md) 了解更多训练技巧
- 阅读 [配置说明](configuration.md) 了解更多配置选项
- 阅读 [数据集结构](dataset-structure.md) 了解更多数据集信息
- 阅读 [常见问题](faq.md) 获取更多解决方案

## 11. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)