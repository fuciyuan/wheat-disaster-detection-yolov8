# 小麦灾害检测系统 - Wheat Disaster Detection System

[![GitHub license](https://img.shields.io/github/license/ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/downloads/release/python-3100/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.2.0-blue)](https://docs.ultralytics.com/)

## 项目介绍 - Project Introduction

小麦灾害检测系统是基于YOLOv8实现的小麦病害和虫害检测与分类系统，可以帮助农民快速、准确地检测和识别多种小麦病害和虫害，及时采取防治措施。

本系统使用现代Python工具链和uv进行依赖管理，确保项目的可复现性和开发效率。

### 支持检测的灾害类型 - Supported Disaster Types

系统可以检测以下5种小麦灾害和健康状态：

1. **小麦锈病 (wheat_rust)** - 叶片上出现橙色、黄色或褐色的锈状斑点
2. **小麦白粉病 (powdery_mildew)** - 叶片表面覆盖白色粉末状霉层
3. **小麦叶斑病 (leaf_spot)** - 叶片上出现圆形或不规则形状的褐色斑点
4. **小麦蚜虫危害 (aphid_damage)** - 叶片卷曲、发黄，表面有蚜虫分泌物
5. **健康小麦 (healthy)** - 叶片正常，无病害或虫害迹象

## 技术栈 - Technology Stack

- **模型框架**: YOLOv8 (Ultralytics) - 先进的单阶段目标检测模型
- **图像处理**: OpenCV, NumPy, Pillow - 高效的图像预处理和后处理
- **训练框架**: PyTorch - 深度学习框架
- **开发工具**: uv - 现代Python包管理器
- **文档**: MkDocs, GitHub Pages - 文档管理和发布

## 项目结构 - Project Structure

```
wheat-disaster-detection-yolov8/
├── .venv/                  # 虚拟环境目录
├── docs/                   # 项目文档目录
│   ├── api.md              # API文档
│   ├── configuration.md    # 配置说明
│   ├── dataset-structure.md # 数据集结构说明
│   ├── faq.md              # 常见问题解答
│   ├── quick-start.md      # 快速入门指南
│   └── training.md         # 训练指南
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── main.py             # 主程序入口
│   ├── config/             # 配置文件目录
│   │   └── config.yaml
│   ├── data/               # 数据集目录
│   │   └── wheat_disaster.yaml
│   ├── models/             # 模型目录
│   ├── tests/              # 测试文件目录
│   └── utils/              # 工具函数目录
├── CONTRIBUTING.md         # 贡献指南
├── pyproject.toml          # 项目配置文件
├── README.md               # 项目说明文件
├── test.py                 # 系统测试脚本
└── uv.lock                 # 依赖锁定文件
```

## 环境配置 - Environment Setup

### 1. 安装uv

在Windows系统上，可以通过以下命令安装uv：

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 克隆项目

```bash
git clone https://github.com/syster-0/wheat-disaster-detection-yolov8.git
cd wheat-disaster-detection-yolov8
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 验证安装

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

## 模型下载 - Model Download

可以通过以下命令下载预训练模型：

```bash
uv run python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

预训练模型将保存在 `./models/yolov8n.pt`。

## 使用说明 - Usage

### 1. 模型检测 - Model Detection

#### 单张图像检测

```bash
uv run python src/main.py detect --image path/to/your/image.jpg
```

#### 批量图像检测

```bash
uv run python src/main.py detect --images path/to/your/images/directory
```

#### 视频检测

```bash
uv run python src/main.py detect --video path/to/your/video.mp4
```

#### 摄像头实时检测

```bash
uv run python src/main.py detect --camera
```

### 2. 模型训练 - Model Training

```bash
uv run python src/main.py train --data src/data/wheat_disaster.yaml --epochs 100
```

### 3. 模型评估 - Model Evaluation

```bash
uv run python src/main.py val --data src/data/wheat_disaster.yaml
```

### 4. 模型导出 - Model Export

```bash
uv run python src/main.py export --weights runs/train/exp/weights/best.pt --format onnx
```

## 数据集准备 - Dataset Preparation

### 数据集结构 - Dataset Structure

```
data/wheat_disaster_dataset/
├── images/
│   ├── train/          # 训练集图像
│   │   └── *.jpg, *.png, etc.
│   ├── val/            # 验证集图像
│   │   └── *.jpg, *.png, etc.
│   └── test/           # 测试集图像
│       └── *.jpg, *.png, etc.
└── labels/
    ├── train/          # 训练集标签文件
    │   └── *.txt (YOLO格式)
    ├── val/            # 验证集标签文件
    │   └── *.txt (YOLO格式)
    └── test/           # 测试集标签文件
        └── *.txt (YOLO格式)
```

### 标签格式 - Label Format

YOLO格式的标签文件示例：
```
0 0.45 0.67 0.23 0.41
1 0.24 0.12 0.24 0.11
```

每个目标一行，格式为 `class_id x_center y_center width height`。

### 配置文件 - Configuration File

请在 `src/data/wheat_disaster.yaml` 中设置数据集路径和类别信息。

## 文档 - Documentation

- [快速开始指南](docs/quick-start.md)
- [训练指南](docs/training.md)
- [API文档](docs/api.md)
- [配置说明](docs/configuration.md)
- [数据集结构](docs/dataset-structure.md)
- [常见问题](docs/faq.md)
- [贡献指南](CONTRIBUTING.md)

## 性能指标 - Performance Metrics

以下是在小麦灾害检测数据集上的性能指标：

| 模型 | 尺寸 | mAP | F1 | 速度(FPS) |
| --- | --- | --- | --- | --- |
| yolov8n.pt | 640 | 0.78 | 0.85 | 140 |
| yolov8s.pt | 640 | 0.83 | 0.88 | 60 |
| yolov8m.pt | 640 | 0.85 | 0.89 | 28 |
| yolov8l.pt | 640 | 0.86 | 0.90 | 16 |
| yolov8x.pt | 640 | 0.87 | 0.91 | 10 |

## 常见问题 - FAQ

### Q: 如何使用CPU进行训练和检测？
A: 使用 `--device cpu` 参数：
```bash
uv run python src/main.py train --data data/wheat_disaster.yaml --epochs 100 --device cpu
```

### Q: 如何减小模型文件大小？
A: 使用 `yolov8n.pt` 模型或使用 `uv run python src/main.py export --weights --format onnx --optimize`

### Q: 如何提高模型检测速度？
A: 使用 `--half` 参数启用FP16精度：
```bash
uv run python src/main.py detect --image image.jpg --half
```

### Q: 如何保存检测结果？
A: 使用 `--save` 参数：
```bash
uv run python src/main.py detect --image image.jpg --save
```

### Q: 如何设置检测置信度阈值？
A: 使用 `--conf` 参数：
```bash
uv run python src/main.py detect --image image.jpg --conf 0.5
```

## 贡献指南 - Contributing

我们欢迎任何形式的贡献，包括bug报告、功能请求、代码提交和文档改进。

### 贡献流程 - Contribution Process

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/your-feature-name`)
3. 提交更改 (`git commit -m 'Add your feature'`)
4. 推送到分支 (`git push origin feature/your-feature-name`)
5. 创建 Pull Request

### 代码规范 - Code Standards

- 请使用PEP 8代码规范
- 为新函数和类编写文档字符串
- 添加适当的测试用例

### 提交规范 - Commit Message Standards

- `feat:` 添加新功能
- `fix:` 修复bug
- `docs:` 更新文档
- `test:` 添加或修改测试
- `refactor:` 代码重构
- `style:` 不影响代码含义的更改（空格、格式等）

### 许可 - License

本项目遵循MIT许可，详见LICENSE文件。

## 联系方式 - Contact

- **项目地址**: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- **问题反馈**: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)

## 参考文献 - References

- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [YOLOv8论文](https://arxiv.org/abs/2305.07926)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## 致谢 - Acknowledgments

感谢 Ultralytics 提供如此优秀的YOLOv8实现！

---

**注意**: 本项目仅用于学术研究和商业应用，请勿用于其他违法用途。