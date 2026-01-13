# 贡献指南

欢迎您贡献代码到本项目！

## 贡献方式

### 提交代码

1. Fork项目到您的GitHub账号
2. 创建功能分支（feature branch）
3. 在分支上开发
4. 提交Pull Request

### 报告问题

通过GitHub Issues提交问题：
- 详细描述问题现象
- 提供复现步骤
- 提供相关截图和日志

### 文档改进

可以通过Pull Request提交文档改进，包括：
- 修正拼写错误
- 完善文档内容
- 添加新文档

## 开发流程

### 1. 克隆项目

```bash
git clone https://github.com/your-username/wheat-disaster-detection-yolov8.git
cd wheat-disaster-detection-yolov8
```

### 2. 创建虚拟环境

```bash
uv venv
uv sync
```

### 3. 运行测试

```bash
python test.py
```

### 4. 开发

创建功能分支：
```bash
git checkout -b feature/your-feature-name
```

进行开发...

### 5. 提交代码

```bash
git add .
git commit -m "Your commit message"
git push origin feature/your-feature-name
```

### 6. Pull Request

创建Pull Request到主分支。

## 代码规范

### 命名规范

- 变量：采用下划线分隔小写字母（例如：image_path, model_config）
- 函数：采用下划线分隔小写字母（例如：load_model, detect_image）
- 类：采用驼峰式命名（例如：WheatModel, ImagePreprocessor）
- 文件名：采用下划线分隔小写字母（例如：main.py, data_loader.py）
- 配置文件：采用yaml格式，使用有意义的命名

### 注释规范

- 每个函数和方法都应该有函数级注释
- 复杂代码块应该添加内部注释
- 使用中文进行注释
- 注释应该清晰说明代码用途

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
    pass
```

### 代码风格

- 使用空格进行缩进
- 每行代码不超过100个字符
- 合理拆分长函数
- 采用面向对象编程风格
- 遵循PEP8规范

## 测试规范

### 单元测试

```python
import unittest

class TestWheatModel(unittest.TestCase):
    def test_model_load(self):
        """测试模型加载"""
        model = load_model()
        self.assertIsNotNone(model)
    
    def test_detect_image(self):
        """测试单张图像检测"""
        model = load_model()
        result = detect_image(model, "examples/test.jpg")
        self.assertIsNotNone(result)
```

### 集成测试

```python
import pytest

def test_inference():
    """模型推理测试"""
    model = load_model()
    img = cv2.imread("examples/test.jpg")
    results = model(img)
    assert len(results) > 0
```

### 性能测试

```python
import time

def test_inference_time():
    """测试推理时间"""
    model = load_model()
    img = cv2.imread("examples/test.jpg")
    start = time.time()
    model(img)
    end = time.time()
    print(f"Inference time: {end - start:.2f}s")
```

## 版本发布

### 版本号

遵循SemVer规范：
- MAJOR：不兼容的API改动
- MINOR：兼容的功能新增
- PATCH：兼容的问题修正

### 发布流程

1. 更新版本号
2. 生成发布说明
3. 上传到GitHub Releases

## 文档规范

### README

README.md应该包括：
- 项目简介
- 安装指南
- 使用方法
- 示例
- 贡献指南

### 文档结构

- 文档放置在docs目录下
- 使用Markdown编写
- 使用清晰的标题层级
- 提供代码示例

```markdown
## 章节标题

### 子章节

内容...

```python
# 代码示例
print("Hello, World!")
```
```

## 分支管理

### 分支策略

- **main**：主分支，发布版本
- **dev**：开发分支，日常开发
- **feature/xxx**：功能分支，开发新功能
- **bugfix/xxx**：修复分支，修复bug
- **hotfix/xxx**：紧急修复分支

### 分支命名规范

- feature/功能名称
- bugfix/修复内容
- hotfix/紧急修复内容

## Pull Request规范

### 标题

使用清晰的标题，例如：
- [Feature] 新增实时视频检测功能
- [Bugfix] 修复内存泄漏问题
- [Docs] 完善训练文档

### 内容

在PR描述中包含：
- 功能说明
- 修改内容
- 测试方法
- 相关截图或示例

### 检查清单

提交PR前需要通过以下检查：
- 代码通过测试
- 代码风格符合规范
- 文档已更新

## 社区参与

### 参与方式

1. 参与GitHub Issue讨论
2. 回答问题和提供帮助
3. 提交PR
4. 改进文档

### 行为准则

- 保持友好
- 提供有建设性的建议
- 尊重他人的工作
- 遵循开源协议

## License

项目采用MIT License，贡献的代码自动归入同一协议。

## 常见问题

### 如何提交一个新功能？

1. 提交Feature Request
2. 讨论功能需求
3. 创建功能分支
4. 开发并测试
5. 提交PR

### 如何修复一个bug？

1. 提交Bug Report
2. 确认问题
3. 创建修复分支
4. 修复并测试
5. 提交PR

### 如何更新文档？

1. 提交文档改进建议
2. 修改文档
3. 提交PR

## 参考资源

- [GitHub Docs](https://docs.github.com/)
- [Git指南](https://guides.github.com/)
- [Python代码规范](https://www.python.org/dev/peps/pep-0008/)
- [Markdown指南](https://www.markdownguide.org/)

## 致谢

感谢所有为本项目做出贡献的开发者。

## 联系方式

- 项目主页：[GitHub](https://github.com)
- 项目邮箱：wheat-disaster@example.com
- 开发团队：team@example.com

---

本贡献指南将持续更新，欢迎随时提供改进建议。