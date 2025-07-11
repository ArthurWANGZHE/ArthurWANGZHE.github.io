---
title: pytorch面经
date: 2025-07-09 10:23:47
cover: /img/Pytorch.png
tags:
---
# Pytorch 面经

PyTorch 是一个开源的机器学习库，广泛应用于计算机视觉和自然语言处理等领域

PyTorch 的两个高级功能

* 强大的GPU加速的张量计算（Numpy）
* 包含自动求导系统的深度神经网络

PyTorch的核心特性

* 动态计算图：使用动态计算图（命令式编程模型），计算图在每次运行的时候都会重新构建。使得模型比较灵活
* 自动微分系统：PyTorch 的 `torch.autograd` 提供了自动微分的功能，可以自动计算导数和梯度

# Pytorch 模块分类以及常用函数

## 1. `torch`（基础张量操作模块）

这是 PyTorch 的底层核心模块，提供类似 NumPy 的张量操作，并支持自动求导。

| 类别      | 常用函数                                                                  |
| --------- | ------------------------------------------------------------------------- |
| 张量创建  | `torch.tensor()`,`torch.zeros()`,`torch.ones()`,`torch.randn()`   |
| 类型转换  | `.float()`,`.long()`,`.to(device)`                                  |
| 数学操作  | `torch.mean()`,`torch.sum()`,`torch.exp()`,`torch.clamp()`        |
| 维度操作  | `.view()`,`.reshape()`,`.permute()`,`.squeeze()`,`.unsqueeze()` |
| 拼接/拆分 | `torch.cat()`,`torch.stack()`,`torch.chunk()`                       |
| 设备控制  | `torch.cuda.is_available()`,`.to('cuda')`,`.cpu()`                  |

## 2. `torch.nn`（神经网络模块）

包含构建神经网络的所有组件：层、损失函数、模型容器等。

| 类别       | 常用类 / 函数                                                             |
| ---------- | ------------------------------------------------------------------------- |
| 层         | `nn.Linear`,`nn.Conv2d`,`nn.LSTM`,`nn.BatchNorm2d`,`nn.Dropout` |
| 激活函数   | `nn.ReLU`,`nn.Sigmoid`,`nn.Tanh`,`nn.LeakyReLU`                   |
| 损失函数   | `nn.CrossEntropyLoss`,`nn.MSELoss`,`nn.BCELoss`,`nn.L1Loss`       |
| 容器       | `nn.Sequential`,`nn.ModuleList`,`nn.ModuleDict`                     |
| 自定义模块 | 继承 `nn.Module`, 实现 `__init__`和 `forward()`                     |

## 3. `torch.nn.functional`（函数式 API）

提供激活函数、loss 函数等的函数式版本，适合在 `forward()` 中灵活调用。

| 类别      | 常用函数                                                        |
| --------- | --------------------------------------------------------------- |
| 激活函数  | `F.relu()`,`F.leaky_relu()`,`F.softmax()`,`F.sigmoid()` |
| 卷积/池化 | `F.conv2d()`,`F.max_pool2d()`                               |
| 损失函数  | `F.cross_entropy()`,`F.mse_loss()`                          |
| dropout   | `F.dropout()`                                                 |

**区别** ：`nn.Module` 是封装好的层；`F.xxx` 是函数，便于组合和自定义逻辑。

## 4. `torch.optim`（优化器模块）

内置多种优化器类，用于更新模型参数。

| 优化器                   | 说明                                              |
| ------------------------ | ------------------------------------------------- |
| `optim.SGD`            | 支持 momentum 和 weight decay                     |
| `optim.Adam`           | 常用，适合大多数情况                              |
| `optim.AdamW`          | L2 正则更标准                                     |
| `optim.lr_scheduler.*` | 学习率调度器，如 `StepLR`,`CosineAnnealingLR` |
| 常用操作                 | `optimizer.step()`,`optimizer.zero_grad()`    |

## 5. `torch.utils.data`（数据加载模块）

提供数据集、数据加载器、自定义数据集等工具。

| 类别       | 常用类                                        |
| ---------- | --------------------------------------------- |
| 数据集     | `torch.utils.data.Dataset`（需继承）        |
| 数据加载器 | `torch.utils.data.DataLoader`               |
| 辅助类     | `random_split`,`Subset`,`ConcatDataset` |

## 6. 训练辅助模块

| 目的            | 工具                                                         |
| --------------- | ------------------------------------------------------------ |
| 保存 / 加载模型 | `torch.save()`,`torch.load()`,`model.state_dict()`     |
| 查看参数数量    | `sum(p.numel() for p in model.parameters())`               |
| 控制计算图      | `torch.no_grad()`,`tensor.detach()`,`requires_grad_()` |
| 梯度裁剪        | `torch.nn.utils.clip_grad_norm_()`                         |

# 损失函数

> 损失函数（Loss Function）用于衡量模型预测值与真实标签之间的差异，是模型“犯错的程度”。

它的值越小，说明模型预测越准确；反之越大，就表示模型偏差大。通过反向传播算法（backpropagation）计算损失函数对模型参数的梯度，从而指导优化器更新模型。

## 常见的损失函数及使用场景

| 损失函数                            | PyTorch 写法               | 适用任务               | 输入要求                | 说明                           |
| ----------------------------------- | -------------------------- | ---------------------- | ----------------------- | ------------------------------ |
| **均方误差 (MSE)**            | `nn.MSELoss()`           | 回归                   | 连续值 vs 连续值        | 常用于预测数值，如房价、坐标等 |
| **交叉熵损失 (CrossEntropy)** | `nn.CrossEntropyLoss()`  | 多分类                 | `logits`+`整数标签` | 自动包含 softmax，常用于分类   |
| **二分类交叉熵 (BCE)**        | `nn.BCELoss()`           | 二分类                 | `概率`+`[0,1]`      | 需要手动使用 `sigmoid`       |
| **BCE with logits**           | `nn.BCEWithLogitsLoss()` | 二分类                 | `logits`+`[0,1]`    | 自动包含 sigmoid，更稳定       |
| **L1 Loss / MAE**             | `nn.L1Loss()`            | 回归                   | 任意连续值              | 更鲁棒，不敏感于离群点         |
| **Huber Loss**                | `nn.SmoothL1Loss()`      | 回归                   | 连续值                  | 兼顾 MSE 与 MAE 的优点         |
| **KL 散度**                   | `nn.KLDivLoss()`         | 分布匹配               | `log_prob`+`prob`   | 常用于蒸馏、分布逼近任务       |
| **CTC Loss**                  | `nn.CTCLoss()`           | 序列对齐（如语音识别） | -                       | 用于无对齐序列标签训练         |
| **NLL Loss**                  | `nn.NLLLoss()`           | 多分类（概率对数）     | `log_softmax`+ 标签   | 与 `log_softmax`搭配使用     |

## 选什么损失函数的基本原则

| 任务类型                  | 推荐损失函数                            |
| ------------------------- | --------------------------------------- |
| 回归问题（数值预测）      | `MSELoss`/`L1Loss`/`SmoothL1Loss` |
| 二分类问题                | `BCEWithLogitsLoss`（强烈推荐）       |
| 多分类（单标签）          | `CrossEntropyLoss`                    |
| 多标签分类（每类独立0/1） | `BCEWithLogitsLoss`                   |
| 蒸馏/生成/分布逼近        | `KLDivLoss`                           |
| 语音/字符对齐             | `CTCLoss`                             |

# 激活函数

> 激活函数是神经网络中作用于每一层（尤其是隐藏层）输出的非线性函数，它决定了模型的**表达能力**和 **收敛速度** 。

如果没有激活函数，多个线性层叠加仍然是线性模型，无法拟合复杂函数。

## 激活函数的作用

1. **引入非线性** ：让网络可以拟合复杂模式（比如图像、语言）
2. **控制信息流动** ：如 ReLU 可将负值置零，稀疏激活
3. **影响梯度传播** ：某些激活函数可能导致梯度消失/爆炸
4. **决定训练速度与收敛效果**

## 常见激活函数汇总（及使用建议）

| 名称                 | 函数表达式                       | PyTorch 写法                  | 特性                             | 适用场景                                  |
| -------------------- | -------------------------------- | ----------------------------- | -------------------------------- | ----------------------------------------- |
| **ReLU**       | `max(0, x)`                    | `nn.ReLU()`or `F.relu(x)` | 稀疏激活，收敛快，简单高效       | 默认首选，适合大多数隐藏层                |
| **Leaky ReLU** | `x if x>0 else αx`            | `nn.LeakyReLU(α)`          | 允许负值通过，缓解“神经元死亡” | ReLU 不收敛时可尝试                       |
| **PReLU**      | `x if x>0 else a*x`（a可学习） | `nn.PReLU()`                | 可学习负斜率                     | 高级调参用                                |
| **Sigmoid**    | `1 / (1 + e^-x)`               | `nn.Sigmoid()`              | 输出在 (0,1)，易饱和             | 二分类输出层，不适合深层网络              |
| **Tanh**       | `(e^x - e^-x)/(e^x + e^-x)`    | `nn.Tanh()`                 | 输出在 (-1,1)，对称              | 曾用于 RNN，现用得少                      |
| **Softmax**    | `exp(xi) / sum(exp(xj))`       | `F.softmax()`               | 将向量映射为概率分布             | 多分类输出层（配合 `CrossEntropyLoss`） |
| **Swish**      | `x * sigmoid(x)`               | `x * torch.sigmoid(x)`      | 平滑、可导、近似线性             | 高端模型如 EfficientNet                   |
| **GELU**       | `x * Φ(x)`（近似正态）        | `nn.GELU()`                 | 用于大模型，如 Transformer       | BERT, GPT 中大量使用                      |
| **ELU**        | `x if x>0 else α*(e^x - 1)`   | `nn.ELU()`                  | 允许负值，更稳定收敛             | 某些任务比 ReLU 好                        |

## PyTorch 中使用激活函数

### 方法 1：模块式（`nn.ReLU()`）

适用于 `nn.Sequential` 或自定义网络中初始化层时：

```python
self.relu = nn.ReLU()
x = self.relu(x)
```

### 方法 2：函数式（`F.relu()`）

适用于 `forward()` 中灵活调用：

```python
import torch.nn.functional as F

x = F.relu(x)
x = F.leaky_relu(x, negative_slope=0.1)

```

## 常见使用场景推荐

| 层/结构                    | 推荐激活函数     | 原因                              |
| -------------------------- | ---------------- | --------------------------------- |
| **隐藏层**           | ReLU / LeakyReLU | 快速、稳定、稀疏                  |
| **RNN单元**          | Tanh / Sigmoid   | 结构限制，经典配置                |
| **Transformer**      | GELU             | 更平滑，BERT 默认                 |
| **二分类输出**       | Sigmoid          | 输出 (0,1)，配合 BCE              |
| **多分类输出**       | Softmax          | 输出为概率分布，配合 CrossEntropy |
| **深层网络无法收敛** | LeakyReLU / ELU  | 解决 ReLU 死亡问题                |

# 优化器

> 优化器负责根据损失函数计算的梯度， **更新模型的参数（weights）** ，从而逐步让模型表现更好。

PyTorch 中的优化器通常来自 `torch.optim` 模块，它们是各种**梯度下降变种算法**的实现。不同优化器的核心区别是： **如何计算梯度、如何调整学习率、是否用动量、是否自适应调整方向** 。

## 常见优化器总结（PyTorch 实现）

| 优化器                             | 类名                              | 特性                                 | 常用场景                         |
| ---------------------------------- | --------------------------------- | ------------------------------------ | -------------------------------- |
| **SGD**                      | `torch.optim.SGD`               | 最基础的随机梯度下降，可加动量       | 小模型、线性分类、强化学习中常见 |
| **SGD + Momentum**           | `torch.optim.SGD(momentum=0.9)` | 增加惯性，减少震荡                   | 比纯 SGD 更稳定                  |
| **Adagrad**                  | `torch.optim.Adagrad`           | 自适应学习率，对稀疏数据友好         | NLP, 词向量训练                  |
| **RMSprop**                  | `torch.optim.RMSprop`           | 类似 Adagrad，但不会让学习率太快衰减 | RNN, LSTM 比较常用               |
| **Adam**                     | `torch.optim.Adam`              | 自适应+动量，训练快，最常用          | 通用首选，CV/NLP皆可             |
| **AdamW**                    | `torch.optim.AdamW`             | 修复了 Adam 的正则缺陷               | Transformer 系模型推荐           |
| **Adadelta / Nadam / RAdam** | …                                | 各种组合优化器                       | 进阶可选                         |
| **LBFGS**                    | `torch.optim.LBFGS`             | 二阶优化器（拟牛顿法）               | 小模型/调参用，不适合大规模      |

## 选择什么优化器的建议

| 情况                    | 推荐优化器            | 原因                    |
| ----------------------- | --------------------- | ----------------------- |
| 通用任务                | `Adam`              | 自动调节学习率 + 收敛快 |
| 模型发散 / 不收敛       | `SGD`+`Momentum`  | 更稳定，控制力强        |
| NLP / 稀疏特征          | `Adagrad`,`AdamW` | 更适合稀疏更新          |
| RNN / LSTM              | `RMSprop`           | 适合处理长序列          |
| Transformer / BERT      | `AdamW`+ 预热学习率 | 原论文推荐配置          |
| 需要可解释性 / 手动控制 | `SGD`               | 更好观察梯度更新过程    |

## 优化器常和调度器配合使用

```python
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()
```

| 调度器                | 功能                         |
| --------------------- | ---------------------------- |
| `StepLR`            | 每隔固定步长降低学习率       |
| `CosineAnnealingLR` | 余弦退火策略                 |
| `ReduceLROnPlateau` | 当 loss 停止下降时降低学习率 |
| `OneCycleLR`        | 适用于快速 warmup            |
| `LambdaLR`          | 自定义调度函数               |

# 模型加速、裁剪、量化

> 模型加速、裁剪、量化是深度学习模型部署和推理优化的重要手段，尤其适用于在 **移动端、嵌入式设备、边缘计算**等资源受限场景中使用 PyTorch 模型。

## 模型加速（Acceleration）

### 方法 1：TorchScript（官方推荐）

```python
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")
```

优点：保留动态图灵活性，运行时更快、更可移植（如部署到 C++、移动端）

### 方法 2：ONNX 导出 + TensorRT

```python
torch.onnx.export(model, dummy_input, "model.onnx", ...)
```

然后用 `onnxruntime` 或 `TensorRT` 部署，可大幅加速推理速度，尤其适合 NVIDIA GPU。

### 方法 3：Operator Fusion

自动将 Conv + BN + ReLU 等合并为一个算子，在 `torch.jit.trace` / `script` 中生效。

## 模型裁剪（Pruning）

> **目的：删除不重要的参数 / 通道，降低模型复杂度**

PyTorch 提供了官方工具包：`torch.nn.utils.prune`

### 示例：对全连接层进行 unstructured pruning

```python
import torch.nn.utils.prune as prune

prune.random_unstructured(model.fc, name='weight', amount=0.3)
```

也可以使用 `L1Unstructured`, `RandomStructured`, `LNStructured` 等方式对整个通道/卷积核裁剪。

### 常见策略

| 类型         | 说明                            |
| ------------ | ------------------------------- |
| Unstructured | 剪单个权重（稀疏）              |
| Structured   | 剪整个通道 / 卷积核（硬件友好） |

---

## 模型量化（Quantization）

> **将 float32 的模型参数和激活转为 int8 / float16 表示，以减少模型大小和加速推理。**

PyTorch 支持三种量化方式：

### 静态量化（Post-Training Static Quantization）

```python
import torch.quantization as quant

model.eval()
model.qconfig = quant.get_default_qconfig("fbgemm")
quant.prepare(model, inplace=True)
# 运行一定量的数据以进行校准
quant.convert(model, inplace=True)
```

适用于部署后不再训练的场景，最常用。

### 动态量化（Dynamic Quantization）

对 RNN / Transformer 类模型友好：

```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 量化感知训练（QAT）

* 最复杂，但效果最好；
* 模型训练时模拟量化误差，最终导出更精度友好的模型；
* 用于对精度非常敏感的部署任务。

## 如何选

| 目的                       | 推荐技术                                        |
| -------------------------- | ----------------------------------------------- |
| 加速部署（Web/移动端）     | `TorchScript`,`ONNX`,`TensorRT`           |
| 模型压缩、训练后导出       | 剪枝 + 静态量化                                 |
| 推理速度瓶颈（全连接/RNN） | 动态量化                                        |
| 精度不能丢 + 部署          | 量化感知训练（QAT）                             |
| Transformer / BERT 模型    | 推荐使用 `AdamW`+ QAT + 融合 LayerNorm/Linear |

# lightning框架

## Lightning 的核心优势

| 优势             | 说明                                                        |
| ---------------- | ----------------------------------------------------------- |
| 更少样板代码     | 不再手写训练循环、验证逻辑                                  |
| 自动管理训练流程 | 包括 `train_step()`、`val_step()`、日志记录、GPU 加速等 |
| 模块化 + 易扩展  | 分离模型、数据、训练策略，便于协作和复用                    |
| 支持分布式训练   | 多卡 / TPU / 混合精度一行代码搞定                           |
| 集成日志工具     | TensorBoard、W&B 等集成开箱即用                             |

## PyTorch Lightning 架构核心类

### `LightningModule`

封装你的模型结构、前向逻辑、loss、optimizer 等。

### `LightningDataModule`

封装 `train_dataloader()`、`val_dataloader()`、`test_dataloader()` 逻辑。

### `Trainer`

类似于 `.fit()` 的总管。

### 使用示例

```python
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch

class LitMLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layer(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

```

# 常见问题

## loss不下降/训练不收敛

### 常见原因及排查建议

| 分类                                 | 可能原因                                     | 解决办法                                                                                |
| ------------------------------------ | -------------------------------------------- | --------------------------------------------------------------------------------------- |
| **学习率问题**                 | 学习率过大：震荡不下降``学习率过小：下降极慢 | ✅ 尝试 `1e-2 ~ 1e-5`的范围调整``✅ 引入 LR Scheduler                                 |
| **数据问题**                   | 标签错、标准化错、数据全为 0 或 1、没打乱    | 🔍 检查数据分布，是否存在泄露、标签错误``✅ 使用 `torchvision.transforms.Normalize()` |
| **模型结构问题**               | 太浅、激活函数用错、BatchNorm 失效           | ✅ 加层、使用适当激活函数，如 `ReLU`、`GELU`✅`model.train()`保证 BN 正常工作     |
| **损失函数问题**               | 类型不匹配（分类用 MSE、回归用 CE）          | ✅ 确保 task ⇋ loss 一致                                                               |
| **优化器问题**                 | 使用不当的优化器参数                         | ✅ 优先尝试 `Adam(lr=1e-3)`✅ SGD 时添加 `momentum=0.9`                             |
| **梯度问题**                   | 梯度消失 / 爆炸，权重更新不动                | 🔍 打印 `param.grad`✅ 尝试 `gradient clipping`或查看梯度分布                       |
| **BatchNorm / Dropout 没关**   | 预测阶段仍在训练模式                         | ✅ 评估时务必 `model.eval()`                                                          |
| **初始化问题**                 | 权重初始化太小 / 全为 0 / 偏差过大           | ✅ 使用 `torch.nn.init`系列，如 `kaiming_uniform_`                                  |
| **正则 / dropout 过强**        | 模型“学不到东西”                           | 🔍 试着去掉 dropout 或减弱 L2 penalty                                                   |
| **过拟合也表现为 loss 上不去** | val loss 很大，train loss 很小               | ✅ 加数据增强、dropout、提前停止（early stop）                                          |

---

### 检查步骤

1. **打印训练 batch 的 Loss** （是否在变）
2. **观察梯度：是否为 None 或全 0**
3. **画出 loss 曲线** ，是否震荡、上升、平坦
4. **尝试过拟合一个小 batch** （比如 batch_size=4，跑100轮）：

> 如果不能拟合一个小 batch，大概率是模型或数据的问题
