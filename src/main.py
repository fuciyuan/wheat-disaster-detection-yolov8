#!/usr/bin/env python3
"""
小麦灾害检测系统主程序
基于YOLOv8实现小麦病害和虫害的检测与分类

Usage:
    python main.py [--image <image_path>] [--video <video_path>] [--camera]
    python main.py train [--data <data_yaml>] [--epochs <epochs>]
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def load_model(model_path: str = "models/yolov8n.pt") -> YOLO:
    """
    加载YOLOv8模型
    
    Args:
        model_path: 模型文件路径
    
    Returns:
        YOLO: 加载后的YOLO模型
    """
    try:
        model = YOLO(model_path)
        print(f"模型加载成功: {model_path}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        raise


def detect_image(model: YOLO, image_path: str) -> None:
    """
    对单张图像进行灾害检测
    
    Args:
        model: YOLO模型
        image_path: 输入图像路径
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 运行检测
    results = model(img)
    
    # 显示结果
    for result in results:
        boxes = result.boxes
        if boxes:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                conf = box.conf[0]
                
                # 绘制边界框
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 绘制类别和置信度
                label = f"{result.names[cls]}: {conf:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow("Wheat Disaster Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="小麦灾害检测系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 检测命令
    detect_parser = subparsers.add_parser("detect", help="进行灾害检测")
    detect_parser.add_argument("--image", type=str, help="输入图像路径")
    detect_parser.add_argument("--video", type=str, help="输入视频路径")
    detect_parser.add_argument("--camera", action="store_true", help="使用摄像头进行实时检测")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data", type=str, default="data/wheat_disaster.yaml", help="数据集YAML文件路径")
    train_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model()
    
    if args.command == "detect":
        if args.image:
            detect_image(model, args.image)
        elif args.video:
            print("视频检测功能将在后续版本中实现")
        elif args.camera:
            print("实时摄像头检测功能将在后续版本中实现")
        else:
            print("请指定检测类型: --image, --video 或 --camera")
    elif args.command == "train":
        print("模型训练功能将在后续版本中实现")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
