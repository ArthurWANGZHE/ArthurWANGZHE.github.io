---
title: Tensorflow面经
date: 2025-07-09 10:23:58
tags:
---
> Tensorflow是一个通过计算图的形式来表述计算的编程系统，计算图也叫数据流图，可以把计算图看做是一种有向图，Tensorflow中的每一个节点都是计算图上的一个Tensor张量，而节点之间的边描述了计算之间的依赖关系(定义时)和数学操作(运算时)。
>
> Tensorflow的关键优势并不在于提供多少的深度神经网络模型，函数或方法，关键的问题是很多神经网络模型从理论上存在，但在实践中，并没有关键的算力去实现它们。

# TensorFlow和Pytorch的区别

| 维度                          | PyTorch                                                    | TensorFlow                                        |
| ----------------------------- | ---------------------------------------------------------- | ------------------------------------------------- |
| **核心风格**            | 动态计算图（eager execution）``写一行，跑一行              | 静态计算图为主（可转 eager）``先定义，后执行      |
| **代码风格**            | 更像 Python，面向开发者                                    | 更像工程框架，偏工业化                            |
| **调试体验**            | 好：可用 `print()`，支持 Python 原生调试                 | 难：Graph 模式下调试不便                          |
| **灵活性**              | 极高，逻辑控制自由                                         | Graph 模式下灵活度受限                            |
| **训练机制**            | 结构清晰，控制粒度高（`train_step`,`loss.backward()`） | 自动化封装度高，使用 `model.fit()`              |
| **部署能力**            | 通过 TorchScript / ONNX / C++ 部署                         | 原生支持 TF Serving / TFLite / TensorRT           |
| **工具生态**            | 有 Lightning、TorchVision、Detectron2、W&B、HuggingFace 等 | 有 Keras、TFX、TPU 支持、TF Hub、TensorBoard 原生 |
| **官方支持**            | Facebook（Meta）                                           | Google                                            |
| **常见应用场景**        | 研究、CV/NLP比赛、RL                                       | 工业部署、大规模系统、移动端优化                  |
| **社区活跃度**          | 非常活跃（特别在 HuggingFace、学术界）                     | 稳定庞大，企业用得多                              |
| **类型提示 / 静态分析** | Pythonic，灵活但容易错                                     | Keras 模型结构更严格，有类型校验                  |


# Tensorflow功能模块


### **基础计算核心模块**

| 模块                | 说明                                                |
| ------------------- | --------------------------------------------------- |
| `tf.Tensor`       | 张量数据结构，类似 NumPy，但支持 GPU 加速和自动求导 |
| `tf.Variable`     | 可训练的变量，用于保存模型参数                      |
| `tf.GradientTape` | 自动微分工具，实现反向传播                          |
| `tf.function`     | 将 Python 函数转为图模式，提高性能                  |
| `tf.math`         | 向量运算、矩阵乘法、广播操作等数学工具              |



### **模型构建与训练模块**

| 模块           | 说明                                                                                 |
| -------------- | ------------------------------------------------------------------------------------ |
| `tf.keras`   | 高层 API，快速构建、训练模型（包含 `layers`,`models`,`losses`,`optimizers`） |
| `tf.nn`      | 低级神经网络原语，如激活函数、卷积、池化等                                           |
| `tf.data`    | 高性能数据输入流水线，支持多线程、shuffle、batch、map                                |
| `tf.train`   | 训练工具（包括 checkpoint、监控、调度器等）                                          |
| `tf.metrics` | 常见评估指标（准确率、MAE、F1 等）                                                   |


### **部署与生产模块**

| 模块                | 说明                                        |
| ------------------- | ------------------------------------------- |
| `tf.saved_model`  | 导出模型为通用部署格式（SavedModel）        |
| `tf.lite`         | 将模型量化并导出为移动端可用格式            |
| `tf.js`           | 将模型导出为 Web 可用（TensorFlow.js）      |
| `tf_serving`      | 用于部署 RESTful API 的模型服务框架         |
| `tfhub.dev`       | TensorFlow Hub：预训练模型共享平台          |
| `TFLiteConverter` | 模型转换工具，从 Keras/TF → TFLite         |
| `tf.distribute`   | 分布式训练工具，支持多 GPU / TPU / 多机训练 |


### **工程化 & 工具链模块**

| 模块                               | 说明                                             |
| ---------------------------------- | ------------------------------------------------ |
| `TensorBoard`                    | 可视化训练过程（loss 曲线、模型结构、embedding） |
| `tf.summary`                     | 日志生成模块（用于 TensorBoard）                 |
| `tf.config`                      | 设置 GPU 显存策略、设备分配                      |
| `tf.random`,`tf.initializers`  | 随机数种子、权重初始化等工具                     |
| `tf.io`,`tf.image`,`tf.text` | 处理数据（图像、文本、TFRecord等）               |


# Tensorflow写CNN

## 加载数据集(以MNIST为例)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 添加通道维度 + 归一化
x_train = x_train[..., tf.newaxis] / 255.0
x_test = x_test[..., tf.newaxis] / 255.0

```


## 构建CNN

```python
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

## 编译模型

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

```

## 训练以及评估

```python
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

```
