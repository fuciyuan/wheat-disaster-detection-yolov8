#!/usr/bin/env python3
"""
小麦灾害检测系统测试脚本
用于验证系统是否可以正常初始化
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

def test_initialization():
    """测试系统初始化"""
    try:
        # 导入主模块
        from main import load_model, main
        print("小麦灾害检测系统初始化成功！")
        
        # 打印使用示例
        print("\n使用示例:")
        print("  检测单张图像: python src/main.py detect --image test.jpg")
        print("  训练模型: python src/main.py train --data data/wheat_disaster.yaml --epochs 100")
        
        return True
    except Exception as e:
        print(f"小麦灾害检测系统初始化失败: {e}")
        return False

if __name__ == "__main__":
    test_initialization()