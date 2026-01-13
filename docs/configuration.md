# 配置说明

## 1. 配置概述

本项目支持多种配置方式，包括：

1. 配置文件（`src/config/config.yaml`）
2. 命令行参数
3. 环境变量

## 2. 配置文件

### 2.1 配置文件结构

配置文件 `src/config/config.yaml` 包含以下部分：

```yaml
# 模型配置
model:
  name: yolov8n.pt
  imgsz: 640
  conf: 0.25
  iou: 0.45

# 训练配置
train:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  save_dir: runs/train

# 数据集配置
dataset:
  path: data/wheat_disaster_dataset
  train: images/train
  val: images/val
  test: images/test
  nc: 5
  names:
    0: rust
    1: powdery_mildew
    2: aphids
    3: grasshopper
    4: drought

# 检测配置
detect:
  device: 0
  save: true
  show: false
  save_txt: true
  save_conf: true

# 数据增强配置
augmentations:
  fliplr: 0.5
  flipud: 0.0
  mosaic: 1.0
  mixup: 0.0

# 性能配置
performance:
  workers: 8
  pin_memory: true
  persistent_workers: true
```

### 2.2 模型配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `model.name` | 模型名称 | string | `yolov8n.pt` |
| `model.imgsz` | 输入图像大小 | int | `640` |
| `model.conf` | 检测阈值 | float | `0.25` |
| `model.iou` | IOU 阈值 | float | `0.45` |

### 2.3 训练配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `train.epochs` | 训练轮数 | int | `100` |
| `train.batch_size` | 批次大小 | int | `16` |
| `train.lr0` | 初始学习率 | float | `0.01` |
| `train.save_dir` | 训练结果保存目录 | string | `runs/train` |

### 2.4 数据集配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `dataset.path` | 数据集路径 | string | `data/wheat_disaster_dataset` |
| `dataset.train` | 训练集路径 | string | `images/train` |
| `dataset.val` | 验证集路径 | string | `images/val` |
| `dataset.test` | 测试集路径 | string | `images/test` |
| `dataset.nc` | 类别数量 | int | `5` |
| `dataset.names` | 类别名称列表 | dict | 5 种灾害类型 |

### 2.5 检测配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `detect.device` | 设备 ID（-1 表示 CPU，0 表示 GPU） | int | `0` |
| `detect.save` | 是否保存检测结果 | bool | `true` |
| `detect.show` | 是否显示检测结果 | bool | `false` |
| `detect.save_txt` | 是否保存检测结果为文本文件 | bool | `true` |
| `detect.save_conf` | 是否保存置信度 | bool | `true` |

### 2.6 数据增强配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `augmentations.fliplr` | 水平翻转概率 | float | `0.5` |
| `augmentations.flipud` | 垂直翻转概率 | float | `0.0` |
| `augmentations.mosaic` | Mosaic 增强概率 | float | `1.0` |
| `augmentations.mixup` | Mixup 增强概率 | float | `0.0` |

### 2.7 性能配置

| 参数 | 描述 | 类型 | 默认值 |
| --- | --- | --- | --- |
| `performance.workers` | 数据加载器工作线程数 | int | `8` |
| `performance.pin_memory` | 是否使用固定内存 | bool | `true` |
| `performance.persistent_workers` | 是否使用持久化工作线程 | bool | `true` |

## 3. 配置文件示例

### 3.1 默认配置

```yaml
# 默认配置
model:
  name: yolov8n.pt
  imgsz: 640
  conf: 0.25
  iou: 0.45

train:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  save_dir: runs/train

dataset:
  path: data/wheat_disaster_dataset
  train: images/train
  val: images/val
  test: images/test
  nc: 5
  names:
    0: rust
    1: powdery_mildew
    2: aphids
    3: grasshopper
    4: drought

detect:
  device: 0
  save: true
  show: false
  save_txt: true
  save_conf: true

augmentations:
  fliplr: 0.5
  flipud: 0.0
  mosaic: 1.0
  mixup: 0.0

performance:
  workers: 8
  pin_memory: true
  persistent_workers: true
```

### 3.2 轻量级配置

```yaml
# 轻量级配置
model:
  name: yolov8n.pt
  imgsz: 320
  conf: 0.5
  iou: 0.5

train:
  epochs: 50
  batch_size: 32
  lr0: 0.001
  save_dir: runs/train

dataset:
  path: data/wheat_disaster_dataset
  train: images/train
  val: images/val
  test: images/test
  nc: 5
  names:
    0: rust
    1: powdery_mildew
    2: aphids
    3: grasshopper
    4: drought

detect:
  device: -1
  save: false
  show: false
  save_txt: false
  save_conf: false

augmentations:
  fliplr: 0.5
  flipud: 0.0
  mosaic: 0.0
  mixup: 0.0

performance:
  workers: 2
  pin_memory: true
  persistent_workers: true
```

### 3.3 高性能配置

```yaml
# 高性能配置
model:
  name: yolov8x.pt
  imgsz: 1024
  conf: 0.25
  iou: 0.45

train:
  epochs: 200
  batch_size: 32
  lr0: 0.01
  save_dir: runs/train

dataset:
  path: data/wheat_disaster_dataset
  train: images/train
  val: images/val
  test: images/test
  nc: 5
  names:
    0: rust
    1: powdery_mildew
    2: aphids
    3: grasshopper
    4: drought

detect:
  device: 0
  save: true
  show: false
  save_txt: true
  save_conf: true

augmentations:
  fliplr: 0.5
  flipud: 0.5
  mosaic: 1.0
  mixup: 0.5

performance:
  workers: 32
  pin_memory: true
  persistent_workers: true
```

## 4. 配置文件加载

### 4.1 加载配置文件

```python
import yaml
from pathlib import Path

def load_config(config_path: str = 'src/config/config.yaml') -> dict:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f'配置文件未找到: {config_file}')
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

# 使用示例
config = load_config()
print(f'加载配置: {config}')
```

### 4.2 保存配置文件

```python
import yaml
from pathlib import Path

def save_config(config: dict, config_path: str = 'src/config/config.yaml') -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False, indent=4, allow_unicode=True)

# 使用示例
config = {
    'model': {
        'name': 'yolov8n.pt',
        'imgsz': 640,
        'conf': 0.25,
        'iou': 0.45,
    },
}
save_config(config)
```

## 5. 配置合并

### 5.1 合并配置文件

```python
def merge_configs(configs: list[dict]) -> dict:
    """
    合并配置文件
    
    Args:
        configs: 配置列表
        
    Returns:
        合并后的配置
    """
    merged_config = {}
    
    for config in configs:
        merged_config.update(config)
    
    return merged_config

# 使用示例
config1 = load_config('src/config/config.yaml')
config2 = {'model': {'imgsz': 1024}}
merged_config = merge_configs([config1, config2])
print(f'合并后的配置: {merged_config}')
```

### 5.2 合并优先级

配置优先级：

1. 命令行参数（最高）
2. 环境变量
3. 配置文件
4. 默认值（最低）

## 6. 配置验证

### 6.1 验证配置

```python
def validate_config(config: dict) -> None:
    """
    验证配置
    
    Args:
        config: 配置字典
    """
    if 'model' not in config:
        raise ValueError('配置中缺少 model 部分')
    
    if 'train' not in config:
        raise ValueError('配置中缺少 train 部分')
    
    if 'dataset' not in config:
        raise ValueError('配置中缺少 dataset 部分')
    
    if 'detect' not in config:
        raise ValueError('配置中缺少 detect 部分')
    
    # 检查模型配置
    if 'name' not in config['model']:
        raise ValueError('配置中缺少 model.name')
    
    if 'imgsz' not in config['model']:
        raise ValueError('配置中缺少 model.imgsz')
    
    if 'conf' not in config['model']:
        raise ValueError('配置中缺少 model.conf')
    
    if 'iou' not in config['model']:
        raise ValueError('配置中缺少 model.iou')
    
    # 检查训练配置
    if 'epochs' not in config['train']:
        raise ValueError('配置中缺少 train.epochs')
    
    if 'batch_size' not in config['train']:
        raise ValueError('配置中缺少 train.batch_size')
    
    if 'lr0' not in config['train']:
        raise ValueError('配置中缺少 train.lr0')
    
    if 'save_dir' not in config['train']:
        raise ValueError('配置中缺少 train.save_dir')
    
    # 检查数据集配置
    if 'path' not in config['dataset']:
        raise ValueError('配置中缺少 dataset.path')
    
    if 'train' not in config['dataset']:
        raise ValueError('配置中缺少 dataset.train')
    
    if 'val' not in config['dataset']:
        raise ValueError('配置中缺少 dataset.val')
    
    if 'nc' not in config['dataset']:
        raise ValueError('配置中缺少 dataset.nc')
    
    if 'names' not in config['dataset']:
        raise ValueError('配置中缺少 dataset.names')

# 使用示例
config = load_config()
validate_config(config)
print('配置验证通过')
```

## 7. 配置文档生成

### 7.1 生成配置文档

```python
def generate_config_docs(config: dict) -> str:
    """
    生成配置文档
    
    Args:
        config: 配置字典
        
    Returns:
        配置文档
    """
    docs = '# 配置文档\n\n'
    
    for section in config:
        docs += f'## {section.capitalize()}\n\n'
        docs += '| 参数 | 描述 | 类型 | 默认值 |\n'
        docs += '| --- | --- | --- | --- |\n'
        
        for key in config[section]:
            value = config[section][key]
            value_type = type(value).__name__
            docs += f'| {section}.{key} | 待添加 | {value_type} | {value} |\n'
        
        docs += '\n'
    
    return docs

# 使用示例
config = load_config()
docs = generate_config_docs(config)
with open('docs/configuration.md', 'w', encoding='utf-8') as f:
    f.write(docs)
print('配置文档生成成功')
```

## 8. 配置模板

### 8.1 配置模板文件

```jinja2
# 配置模板
model:
  name: {{ model_name|default('yolov8n.pt') }}
  imgsz: {{ imgsz|default(640) }}
  conf: {{ conf|default(0.25) }}
  iou: {{ iou|default(0.45) }}

train:
  epochs: {{ epochs|default(100) }}
  batch_size: {{ batch_size|default(16) }}
  lr0: {{ lr0|default(0.01) }}
  save_dir: {{ save_dir|default('runs/train') }}

dataset:
  path: {{ dataset_path|default('data/wheat_disaster_dataset') }}
  train: {{ train_path|default('images/train') }}
  val: {{ val_path|default('images/val') }}
  test: {{ test_path|default('images/test') }}
  nc: {{ nc|default(5) }}
  names:
    0: rust
    1: powdery_mildew
    2: aphids
    3: grasshopper
    4: drought
```

### 8.2 生成配置文件

```python
import jinja2
from pathlib import Path

def generate_config(template_path: str, **kwargs) -> dict:
    """
    生成配置文件
    
    Args:
        template_path: 模板文件路径
        **kwargs: 模板参数
        
    Returns:
        生成的配置
    """
    template_file = Path(template_path)
    
    if not template_file.exists():
        raise FileNotFoundError(f'模板文件未找到: {template_file}')
    
    template = jinja2.Template(template_file.read_text(encoding='utf-8'))
    config_text = template.render(**kwargs)
    
    import yaml
    config = yaml.safe_load(config_text)
    
    return config

# 使用示例
config = generate_config('src/config/template.j2', model_name='yolov8m.pt', imgsz=1024)
save_config(config, 'src/config/config.yaml')
print('配置文件生成成功')
```

## 9. 配置管理工具

### 9.1 配置管理脚本

```python
#!/usr/bin/env python3
import argparse
import yaml
from pathlib import Path

def main():"""主函数"""
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('--load', help='加载配置文件')
    parser.add_argument('--save', help='保存配置文件')
    parser.add_argument('--merge', nargs='+', help='合并配置文件')
    parser.add_argument('--validate', help='验证配置文件')
    parser.add_argument('--docs', help='生成配置文档')
    
    args = parser.parse_args()
    
    if args.load:
        config = load_config(args.load)
        print(f'加载配置: {args.load}')
        print(yaml.dump(config, sort_keys=False, indent=4, allow_unicode=True))
    
    elif args.save:
        config = yaml.safe_load(input('输入配置 JSON: '))
        save_config(config, args.save)
        print(f'保存配置: {args.save}')
    
    elif args.merge:
        configs = []
        for config_file in args.merge:
            config = load_config(config_file)
            configs.append(config)
        merged_config = merge_configs(configs)
        print(yaml.dump(merged_config, sort_keys=False, indent=4, allow_unicode=True))
    
    elif args.validate:
        config = load_config(args.validate)
        validate_config(config)
        print('配置验证通过')
    
    elif args.docs:
        config = load_config(args.docs)
        docs = generate_config_docs(config)
        print(docs)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
```

### 9.2 配置管理脚本使用

```bash
# 加载配置
python config.py --load src/config/config.yaml

# 保存配置
python config.py --save src/config/config.yaml

# 合并配置
python config.py --merge src/config/config1.yaml src/config/config2.yaml

# 验证配置
python config.py --validate src/config/config.yaml

# 生成配置文档
python config.py --docs src/config/config.yaml > docs/configuration.md
```

## 10. 下一步

- 阅读 [快速开始](quick-start.md) 了解基本使用方法
- 阅读 [训练指南](training.md) 了解更多训练技巧
- 阅读 [API 文档](api.md) 了解更多接口
- 阅读 [数据集结构](dataset-structure.md) 了解数据集格式
- 阅读 [FAQ](faq.md) 了解常见问题

## 11. 联系方式

如有任何问题，可以通过以下方式联系我们：

- 项目地址: [https://github.com/syster-0/wheat-disaster-detection-yolov8](https://github.com/syster-0/wheat-disaster-detection-yolov8)
- 问题反馈: [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues)