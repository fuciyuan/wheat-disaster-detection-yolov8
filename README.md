# Wheat Disaster Detection with YOLOv8

基于YOLOv8的小麦灾害检测系统，用于检测和分类5种主要小麦灾害：
1. **锈病 (Rust)** - 小麦常见真菌病害
2. **白粉病 (Powdery Mildew)** - 叶片出现白色粉末状
3. **蚜虫 (Aphid)** - 小型昆虫害虫
4. **小麦瘟病 (Wheat Blast)** - 毁灭性真菌病害
5. **健康 (Healthy)** - 健康小麦

## 技术栈

* **YOLOv8** - 最新一代目标检测模型
* **ultralytics** - YOLOv8官方实现
* **OpenCV** - 计算机视觉库
* **NumPy** - 数值计算库
* **Python 3.13.6+** - 编程语言
* **uv** - Python项目管理工具

## 文档

- **快速开始**: [docs/quick-start.md](docs/quick-start.md)
- **训练指南**: [docs/training.md](docs/training.md)
- **API文档**: [docs/api.md](docs/api.md)
- **配置说明**: [docs/configuration.md](docs/configuration.md)
- **数据集结构**: [docs/dataset-structure.md](docs/dataset-structure.md)
- **常见问题**: [docs/faq.md](docs/faq.md)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 目录结构

```
wheat-disaster-detection-yolov8/
├── src/                      # 源代码
│   ├── main.py              # 主入口
│   ├── config/
│   │   └── config.yaml
│   ├── data/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── tests/
│       └── __init__.py
├── data/                     # 数据集
│   ├── wheat_disaster.yaml  # 数据集配置
│   ├── train/
│   ├── val/
│   └── test/
├── models/                   # 模型文件
├── docs/                     # 文档
│   ├── quick-start.md
│   ├── training.md
│   ├── api.md
│   ├── configuration.md
│   ├── dataset-structure.md
│   └── faq.md
├── tests/                    # 测试代码
├── examples/                 # 示例代码
├── .github/                  # GitHub配置
├── .gitignore
├── pyproject.toml            # 项目配置
├── README.md
├── CONTRIBUTING.md           # 贡献指南
└── test.py                   # 测试脚本
```

## 快速开始

### 环境配置

1. 安装uv：
   ```bash
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. 安装依赖：
   ```bash
   uv sync
   ```

3. 运行测试：
   ```bash
   python test.py
   ```

### 模型下载

1. 下载YOLOv8预训练模型：
   ```bash
   python src/main.py download --model yolov8n.pt
   ```

2. 或者使用自定义训练好的模型

### 运行检测

#### 单张图片检测
```bash
python src/main.py detect --image path/to/image.jpg
```

#### 视频检测
```bash
python src/main.py detect --video path/to/video.mp4
```

#### 实时摄像头检测
```bash
python src/main.py detect --camera
```

### 模型训练

```bash
python src/main.py train --data data/wheat_disaster.yaml --epochs 100 --batch 16
```

## 数据集结构

### 推荐结构
```
data/
├── wheat_disaster.yaml    # 数据集配置
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

## 配置文件说明

配置文件位于 `src/config/config.yaml`，包含:
- **model**：模型配置（尺寸、置信度阈值等）
- **train**：训练参数（轮数、批次大小等）
- **dataset**：数据集路径和类别

## 贡献指南

欢迎提交Issue和Pull Request。请遵循以下规范：
1. 提交代码前确保所有测试通过
2. 添加详细的注释和文档
3. 保持代码风格一致
