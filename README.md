# HBO Classification: Classroom NeuroAI Analysis

这是一个基于多模态数据（fNIRS 脑成像数据 + 课堂对话文本）的深度学习项目，旨在识别学生在自然课堂互动中的 **“互动形式”** 和 **“认知层级”**。

## 🧠 项目简介

本项目采用 **双流 Transformer (Dual-Stream Transformer)** 架构：

- **Brain Stream**: 使用 Transformer Encoder 处理 44 通道 fNIRS 时间序列数据。
- **Text Stream**: 使用预训练 BERT 模型处理同步的对话文本。
- **Fusion**: 多模态特征融合后，进行 **12分类多标签 (Multi-label)** 预测。

**支持的标签体系 (12类)：** 包括基础知识、个人观点、比较归纳、分析阐释、迁移创新、拓展建构等 6 个认知层级下的“提问”与“回应”。

------

## 🛠️ 环境安装 (Installation)

推荐使用 Conda 管理环境。

### 1. 创建并激活环境

Bash

```shell
# 创建环境 (建议 python 3.8 或 3.9)
conda create -n hbo python=3.9

# 激活环境
conda activate hbo
```

### 2. 安装依赖

请确保项目根目录下有 `requirements.txt`。

Bash

```shell
pip install -r requirements.txt
```

*(主要依赖: torch, pandas, numpy, transformers, scikit-learn, tqdm, openpyxl)*

------

## 🚀 快速运行 (Usage)

本项目支持 **Mac (MPS)**、**NVIDIA (CUDA)** 和 **CPU** 自动切换。

代码已更新，支持灵活指定数据集路径（文件夹或单文件）以及自定义模型保存位置。

### 1. 训练 Child 数据 (批量模式)

读取 `./data/child` 文件夹下的所有数据，并将模型保存为 `child_model.pth`。

```Bash
python train.py --data_path ./data/child --model_name child_model.pth
```

### 2. 训练 Adult 数据 (批量模式)

读取 `./data/adult` 文件夹下的所有数据，并将模型保存为 `adult_model.pth`。

```Bash
python train.py --data_path ./data/adult --model_name adult_model.pth
```

### 3. 指定单文件模式 (调试用)

如果你只想针对某一个特定的 `.xlsx` 文件进行调试或训练：

```Bash
python train.py --data_path "./data/adult/experiment_001.xlsx" --model_name debug_model.pth
```

### 4. 参数说明

| **参数**       | **默认值**       | **说明**                                                     |
| -------------- | ---------------- | ------------------------------------------------------------ |
| `--data_path`  | `./data/child`   | **核心参数**。可以是包含xlsx的**文件夹路径**，也可以是**单个文件路径**。 |
| `--save_dir`   | `./model_save`   | 模型保存的文件夹，若不存在会自动创建。                       |
| `--model_name` | `best_model.pth` | 保存的模型文件名（只保存验证集 F1 最高的模型）。             |

------

## 📂 数据格式说明 (Data Format)

数据文件需为 `.xlsx` 格式，且必须包含以下两个 Sheet：

1. **`utterance` (文本流)**
   - 包含对话内容 (`content`)、标签 (`label`)。
   - **关键列**: `startRow`, `endRow` (用于索引对应的脑电数据行号)。
2. **`hbo` (脑电流)**
   - 纯数值矩阵，前 44 列为 fNIRS 通道数据。

------

## ⚙️ 项目结构 (Structure)

Plaintext

```
hbo_classification/
├── data/                  # 存放 .xlsx 数据文件
│   └── S12_C31_黄子硕.xlsx
├── src/
│   ├── config.py          # [配置文件] 修改超参数、路径、显卡设置
│   ├── dataset.py         # [数据加载] 处理Excel读取、对齐、重采样增强
│   └── model.py           # [模型定义] 双塔 Transformer 网络结构
├── main.py                # [主程序] 训练、验证、参数解析入口
├── requirements.txt       # 依赖列表
└── README.md              # 项目说明文档
```

## 🔧 配置与调优

如果你需要修改训练参数（如 `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`），请直接编辑 `src/config.py` 文件。

- **数据增强**: 训练模式下，代码会自动对稀有类别（如迁移创新、拓展建构）进行 **Oversampling (重采样)** 以缓解类别不平衡。
- **防止泄漏**: 代码已实现严格的“物理隔离切分”，确保验证集数据从未参与过训练或重采样。

------

## 📝 常见问题

**Q: 运行日志里显示的 `MPS` 是什么意思？**

- **A**: `MPS` (Metal Performance Shaders) 是 MacOS 系统（特别是搭载 **M1/M2/M3/M4** 芯片的 Mac）专用的 **GPU 加速模式**。
  - 本项目代码内置了智能硬件检测：
    - 如果你是用 **NVIDIA 显卡** (Windows/Linux)，会自动使用 `CUDA`。
    - 如果你是用 **Mac**，会自动使用 `MPS` (训练速度通常比 CPU 快 5-10 倍)。
    - 如果都没有，则回退到 `CPU`。

**Q: 为什么验证集的 F1 分数波动很大（甚至有时很低）？**

- **A**: 这是由数据量和评估方式决定的正常现象：
  1. **样本稀缺**: 单个学生的验证集通常只有 20-30 条数据。在多标签分类中，只要预测错 1-2 条，F1 分数就会剧烈抖动。
  2. **物理隔离**: 为了防止数据泄漏，我们严格执行了“先切分、后增强”的策略。验证集是模型从未见过的“生数据”，没有任何数据增强的加持，因此分数反映了模型真实的（且严苛的）泛化能力。
  3. **建议**: 尽量将更多同学的 `.xlsx` 文件放入 `data/` 目录进行 **联合训练**。随着数据量的增加，F1 指标会逐渐趋于稳定和上升。

**Q: 为什么日志显示的“训练集大小”比原始文件行数多很多？**

- **A**: 这是因为开启了 **Oversampling (重采样增强)**。
  - 为了解决类别不平衡问题（例如“基础知识”很多，“迁移创新”很少），代码会自动识别训练集中的稀有样本，并将其复制 5 倍。
  - **注意**: 这种增强 **仅针对训练集**，验证集永远保持原始大小和分布，以确保评估的公正性。