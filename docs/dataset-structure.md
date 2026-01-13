# Wheat Disaster Detection Dataset Structure

## 数据集推荐结构

```
data/
├── wheat_disaster.yaml    # 数据集配置文件
├── train/
│   ├── images/
│   │   ├── img0001.jpg
│   │   └── img0002.jpg
│   └── labels/
│       ├── img0001.txt
│       └── img0002.txt
├── val/
│   ├── images/
│   │   └── ...
│   └── labels/
│       └── ...
└── test/
    ├── images/
    └── labels/
```

## 数据集配置文件 (wheat_disaster.yaml)

```yaml
path: ../data/wheat_disaster
train: train/images
val: val/images
test: test/images

nc: 5
names: ['rust', 'powdery_mildew', 'aphid', 'wheat_blast', 'healthy']
```

## 标注格式

使用YOLO格式进行标注，每个标签文件包含以下格式：
```
<class_id> <x_center> <y_center> <width> <height>
```

* `class_id`: 类别编号 (0-4)
* `x_center`: 目标中心x坐标 (归一化)
* `y_center`: 目标中心y坐标 (归一化)
* `width`: 目标宽度 (归一化)
* `height`: 目标高度 (归一化)