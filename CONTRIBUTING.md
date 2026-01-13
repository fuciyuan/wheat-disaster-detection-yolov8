# 贡献指南

## 欢迎

欢迎您为小麦灾害检测系统项目做出贡献！无论您是想报告错误、提出新功能建议，还是提交代码，我们都非常欢迎您的参与。

## 贡献方式

### 1. 报告错误

如果您在使用过程中发现错误，请按照以下步骤报告：

1. 在 [GitHub Issues](https://github.com/syster-0/wheat-disaster-detection-yolov8/issues) 中搜索类似的错误
2. 如果没有找到类似错误，请创建一个新的 Issue
3. 在 Issue 中详细描述错误现象和重现步骤
4. 提供相关的截图或日志文件

### 2. 提出新功能建议

如果您有新的功能建议，可以在 [GitHub Discussions](https://github.com/syster-0/wheat-disaster-detection-yolov8/discussions) 中分享您的想法：

1. 选择 "Ideas" 板块
2. 描述您的功能建议
3. 说明为什么该功能有用

### 3. 改进文档

文档是项目的重要组成部分。如果您发现文档有错误或缺失，可以提交 Pull Request 进行改进。

### 4. 提交代码

如果您想为项目贡献代码，可以按照以下步骤操作：

1. Fork 本项目
2. 创建功能分支
3. 提交您的更改
4. 创建 Pull Request

## 开发流程

### 1. 克隆项目

```bash
git clone https://github.com/syster-0/wheat-disaster-detection-yolov8.git
cd wheat-disaster-detection-yolov8
```

### 2. 创建虚拟环境

```bash
uv sync
```

### 3. 运行测试

```bash
uv run python test.py
```

### 4. 开发

在您进行开发之前，请确保您的代码符合项目的代码规范。

### 5. 提交代码

```bash
git commit -m "描述您的更改"
```

### 6. 推送分支

```bash
git push origin feature/your-feature-name
```

### 7. 创建 Pull Request

在 GitHub 上创建 Pull Request，详细描述您的更改。

## 代码规范

### 1. 命名规范

- 使用小写字母和下划线作为函数和变量名，例如: `load_model()`, `image_path`
- 使用 PascalCase 作为类名，例如: `WheatDisasterDetector`
- 使用大写字母和下划线作为常量，例如: `MAX_IMAGE_SIZE`, `DEFAULT_CONF_THRESHOLD`

### 2. 注释规范

- 为每个函数和类编写文档字符串
- 注释应该清晰易懂，解释代码的作用和原理
- 避免不必要的注释

### 3. 代码风格

- 遵循 PEP 8 代码规范
- 使用 4 个空格进行缩进
- 每行代码不超过 120 个字符
- 代码结构清晰，逻辑分明

## 测试规范

### 1. 单元测试

所有新功能都应该有对应的单元测试。单元测试应该覆盖基本的功能和边界条件。

### 2. 集成测试

在提交 Pull Request 之前，请确保所有集成测试通过。

### 3. 性能测试

对于涉及性能的更改，请提供性能测试结果。

## 版本发布

我们遵循语义化版本规范：

- 主版本号: 重大更改
- 次版本号: 新增功能
- 修订版本号: 错误修复

## 文档规范

### 1. 结构清晰

文档应该有清晰的结构，使用适当的标题和段落。

### 2. 内容准确

文档内容应该准确无误，反映最新的代码实现。

### 3. 语言规范

文档使用中英文双语，但中文内容应占主导。

## 分支管理

- `main`: 主分支，包含稳定的代码
- `develop`: 开发分支，包含下一个版本的功能
- `feature/*`: 功能分支，用于开发新功能
- `bugfix/*`: 修复分支，用于修复错误

## Pull Request 规范

### 1. 标题

Pull Request 的标题应该清晰明了，描述更改的主要内容。

### 2. 描述

在 Pull Request 的描述中，应该包含以下信息：

- 更改的目的
- 更改的内容
- 测试结果
- 相关的 Issue

### 3. 检查清单

在提交 Pull Request 之前，请确保：

- 代码通过了所有测试
- 代码符合项目的代码规范
- 添加了必要的注释和文档
- 提交信息清晰明了
