---
title: pytorch面经
date: 2025-07-09 10:23:47
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

# 优化器

# 模型加速、裁剪、量化

# lightning框架

# 常见问题

## loss不下降

## 训练不收敛
