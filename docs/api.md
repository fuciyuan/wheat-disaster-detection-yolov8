# API Documentation

## 主程序接口

### main.py

### 主入口函数

```python
def main() -> None:
    """
    程序主入口
    
    Usage:
        python main.py detect --image path/to/image.jpg
        python main.py train --data data.yaml --epochs 100
    """
```

### load_model

```python
def load_model(model_path: str = "models/yolov8n.pt") -> YOLO:
    """
    加载YOLOv8模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        YOLO: 加载后的YOLO模型
    
    Raises:
        Exception: 当无法加载模型时
    """
```

### detect_image

```python
def detect_image(model: YOLO, image_path: str) -> None:
    """
    对单张图像进行灾害检测
    
    Args:
        model: YOLO模型实例
        image_path: 输入图像的路径
    
    Returns:
        None
    """
```

## 命令行参数

### 检测命令

```bash
python main.py detect
```

参数：
- `--image <path>`：单张图像的路径
- `--video <path>`：视频文件路径
- `--camera`：使用摄像头进行实时检测

### 训练命令

```bash
python main.py train
```

参数：
- `--data <path>`：数据集配置文件路径 (default: data/wheat_disaster.yaml)
- `--epochs <num>`：训练轮数 (default: 100)
- `--batch <num>`：批次大小 (default: 16)
- `--lr0 <float>`：初始学习率 (default: 0.01)
- `--save_dir <path>`：结果保存路径 (default: runs/train)

### 验证命令

```bash
python main.py val
```

参数：
- `--data <path>`：数据集配置文件路径
- `--weights <path>`：模型权重文件路径

### 导出命令

```bash
python main.py export
```

参数：
- `--weights <path>`：模型权重文件路径
- `--format <format>`：导出格式 (default: onnx)

## 数据结构

### 检测结果

每个检测结果包含：
- `boxes`：边界框坐标（x1, y1, x2, y2）
- `conf`：置信度
- `cls`：类别ID
- `names`：类别名称

### 示例结果

```python
results = model(img)
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        name = result.names[cls]
        
        print(f"{name}: {conf:.2f}, ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
```

## 配置文件API

### config.yaml结构

```yaml
model:
  name: "yolov8n"  # 模型名称
  imgsz: 640      # 输入图像大小
  conf: 0.25      # 置信度阈值
  iou: 0.45       # IOU阈值

train:
  epochs: 100     # 训练轮数
  batch_size: 16  # 批次大小
  lr0: 0.01       # 初始学习率
  save_dir: "runs/train"  # 结果保存目录

dataset:
  path: "data"      # 数据集根目录
  train: "train/images"  # 训练集路径
  val: "val/images"      # 验证集路径
  test: "test/images"    # 测试集路径
  nc: 5             # 类别数量
  names: ['rust', 'powdery_mildew', 'aphid', 'wheat_blast', 'healthy']  # 类别名称
```

## 模型API

### 自定义模型类

```python
class WheatModel:
    def __init__(self, model_path: str = "models/yolov8n.pt"):
        """初始化模型
        
        Args:
            model_path: 模型文件路径
        """
        pass
    
    def predict(self, image: np.ndarray) -> list:
        """对图像进行预测
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            list: 检测结果列表
        """
        pass
    
    def train(self, data_config: str, epochs: int = 100) -> None:
        """训练模型
        
        Args:
            data_config: 数据集配置文件路径
            epochs: 训练轮数
        """
        pass
    
    def validate(self, data_config: str) -> dict:
        """验证模型
        
        Args:
            data_config: 数据集配置文件路径
            
        Returns:
            dict: 验证结果
        """
        pass
```

## 错误处理

### 常见错误

1. 无法读取图像：
   ```python
   FileNotFoundError: Could not find image: path/to/image.jpg
   ```

2. 模型加载失败：
   ```python
   ValueError: Model file not found: models/yolov8n.pt
   ```

3. 数据集格式错误：
   ```python
   FileNotFoundError: Could not find data.yaml: data/wheat_disaster.yaml
   ```

### 异常捕获

```python
try:
    model = load_model()
    detect_image(model, image_path)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

## 性能指标

### 评估指标

- **AP (Average Precision)**: 平均精度
- **mAP50**: IoU阈值为0.50的平均精度
- **mAP50-95**: IoU阈值为0.50到0.95的平均精度
- **F1 Score**: F1得分
- **Recall**: 召回率
- **Precision**: 准确率

### 示例评估结果

```
Epoch   1/100
Train:  box_loss=0.0201, cls_loss=0.0102, dfl_loss=0.0153
Val:    box_loss=0.0203, cls_loss=0.0105, dfl_loss=0.0152
Results saved to runs/train/exp

Results:
mAP50: 0.985, mAP50-95: 0.823
Recall: 0.978, Precision: 0.989
```

## 部署API

### 示例Flask部署

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from main import load_model

app = Flask(__name__)
model = load_model()

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```