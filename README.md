# 基于 Food-101 数据集的食物识别系统（课程大作业）

本项目实现了一个基于 Food-101 数据集的深度学习食物分类系统。支持模型训练与自主图像验证，用于人工智能导论大作业。

## 项目结构

```
.
├── data/
│   └── food-101/               # 存放原始数据集（需自行下载）
├── test_images/                # 自主验证图像及标注文件
│   ├── *.jpg / *.png           # 待测试的食物图片
│   └── labels.json       # 图片对应的真实类别标注
├── train.py                    # 模型训练脚本
├── test.py                     # 模型测试脚本
├── requirements.txt            # Python 依赖列表
└── README.md                   # 项目说明文档
```

## 环境配置
1.安装numpy等库
```bash
pip install -r requirements.txt
```

2.安装PyTorch with CUDA support
若使用 Windows 系统，可运行提供的 `.bat` 批处理文件完成依赖安装

## 数据集准备

本实验使用 [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) 数据集，包含 101 类食物，共 101,000 张图片。

**下载与部署步骤：**

1. 从官方链接下载数据集压缩包：  
   [https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
2. 解压后，将 `food-101` 文件夹整体放入项目根目录下的 `data/` 文件夹中。
   - 最终路径应为：`./data/food-101/`
3. 由于数据集过大（约 5GB），未上传至 GitHub 仓库。如需快速获取，请联系作者获取 ZIP 文件。

## 模型训练

完成数据准备后，执行以下命令开始训练：

```bash
python train.py
```

训练过程中会自动加载数据、构建模型（默认使用 ResNet50 ），并保存最优模型权重。您可以根据需要修改训练参数（如批次大小、学习率、训练轮数等）。

## 自主图像验证

### 1. 构建自定义测试集

将您自己的食物图像放置于 `test_images/` 目录下，并创建名为 `labels.json` 的标注文件。

**文件命名规则**：  
图像文件名可任意（如 `apple_pie_01.jpg`），但 `labels.json` 中的类别**必须**为 Food-101 数据集内的官方类别名称（例如 `apple_pie` 等）。

**标注文件格式示例** (lables.json`)：

```json
{
    "apple_pie_01.jpg": "apple_pie",
    "burger_02.jpg": "hamburger",
    "sushi_03.png": "sushi"
}
```

> **注意**：本项目不强制要求文件名与类别名保持一致，但 `json` 中的键必须与图片文件名完全匹配（含扩展名）。

### 2. 执行测试

在终端中运行：

```bash
python test.py
```

脚本将加载训练好的模型，对 `test_images/` 下的每张图片进行分类预测，并与 `labels.json` 中的真实标签进行比对，输出准确率及每张图片的预测结果。

## 版权与致谢

- 测试所用的数据库来源于网络，仅用于课程作业。如有版权问题，请联系作者删除。
- Food-101 数据集版权归原始发布者所有，使用请遵守其许可协议。
---

有任何疑问欢迎联系作者。祝项目顺利！

--- 
