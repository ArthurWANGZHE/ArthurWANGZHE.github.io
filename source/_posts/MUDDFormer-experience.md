---
title: MUDDFormer论文阅读/复现
date: 2025-07-11 11:00:05
tags:
---
> **提出了一个新的连接机制** ，可看作是对 Transformer 结构的一种 **通用、兼容性强、性能更优的替代设计** 。

# 几种连接方式对比

## 传统连接方式（Fully Sequential）

每一层的输出只传递给下一层作为输入

```python-repl
X₀ = Embedding(input)
X₁ = Layer₁(X₀)
X₂ = Layer₂(X₁)
...
X_L = Layer_L(X_{L-1})
```

* **顺序传递** ，层与层之间只有一个“输入→输出”通道。
* 网络变深时，容易出现：
  * **梯度消失** （训练难）
  * **表示退化** （深层特征几乎没有新信息）


* 跨层信息流被限制在“串行通道”中，远层信息难以传回浅层。
* 不适合训练非常深的网络。


## 残差连接方式（Residual Connection, ResNet）

每层除了计算输出,还会直接把输入加到输出上,transfomer中还会加上LayerNorm

```python-repl
X₁ = Layer₁(X₀) + X₀
X₂ = Layer₂(X₁) + X₁
...

X_norm = LayerNorm(X)
X_out = SubLayer(X_norm) + X
```


* **跳跃连接** ：允许梯度从深层“跳回”浅层， **缓解梯度消失** 。
* **恒等路径** ：如果某一层没学到什么，那至少还能“保留原输入”。
* **训练更稳定** 、 **收敛更快** 。


* 所有信息最终都压在一条 residual stream（残差流）上传递，容易过载。
* 层数很深时，会出现  **representation collapse** （每层的输出特征越来越相似，信息熵下降）。
* 没有“多源信息聚合”：每层只能依赖前一层的特征。


## MUDD连接方式（Multiway Dynamic Dense）

设计目标是突破"单一残差流",总结有3大特点

**Dense（密集连接）**

每层都能访问 **所有前面层的输出** ：

```python-repl
X_i = Aggregate(X₀, X₁, ..., X_{i-1})
```

每层不是只接收 $X_{i-1}$，而是从所有 $X_j（j ≤ i-1）$中**聚合**信息。

**Dynamic（动态权重）** 

连接权重不是固定的，而是由当前隐藏状态 X_i  **动态生成** ：

* 对每个序列位置都单独生成权重
* 更细粒度、更灵活的特征融合方式

**Multiway（多路流）** 

将输入拆分成：

* Query stream（Q）
* Key stream（K）
* Value stream（V）
* Residual stream（R）

每个 stream 都有独立的密集动态聚合

```python-repl
X_Qᵢ = DA_Qᵢ(X₀, ..., Xᵢ)
X_Kᵢ = DA_Kᵢ(X₀, ..., Xᵢ)
X_Vᵢ = DA_Vᵢ(X₀, ..., Xᵢ)
X_Rᵢ = DA_Rᵢ(X₀, ..., Xᵢ)
```

![1752203366868](image/MUDDFormer-experience/1752203366868.png)

# muddformer

```python-repl
class MultiwayDynamicDenseBlock(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx: int, last_layer=False) -> None:
        super().__init__()
        self.norm = RMSnormNoscale(epsilon=config.norm_eps)
        self.C = len(config.dense_type) if not last_layer else 1
        self.lidx = lidx
        l = lidx + 2
        hid_dim, out_dim = l * self.C, l * self.C
        if last_layer and config.expand_last: hid_dim *= 4  
        if config.round64: hid_dim = (hid_dim// 64 +1) * 64 
        self.w1 = nn.Linear(config.dim, hid_dim, bias=False)
        self.act = nn.GELU() 
        self.w2 = nn.Linear(hid_dim, out_dim, bias=False)
  
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x) 
        dw = self.w2(self.act(self.w1(x))) # BTD->BTL
        dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.C)
        return dw
  
    def layer_mix(self, hids, dw)-> Tensor:
        x = tuple([sum(dw[cidx,:,:,j,None] * hids[j] for j in range(self.lidx+2)) for cidx in range(self.C)]) # BTL, LBTD-> BTD
        return x
```


* 输入：当前层输出 `x`（B, T, D）
* 输出：动态生成的 dense 聚合权重 `dw`，形状为 `(C, B, T, L)`，表示：
  * `C`：多路（通常是 Q/K/V/R）
  * `L`：历史层数量
  * `B, T`：每个位置每个样本独立动态生成权重

```terminal
Start generating tokens, but it will take a few minutes to compile at the first time.
Generated text: Beijing is the capital of China. London is the capital of England.

The capital of the United States
Time consumed at iteration 0: 1.5410542488098145s
Time consumed at iteration 1: 1.3622238636016846s
Time consumed at iteration 2: 1.3594660758972168s
Time consumed at iteration 3: 1.369415283203125s
Time consumed at iteration 4: 1.3852148056030273s
Time consumed at iteration 5: 1.3903260231018066s
Time consumed at iteration 6: 1.358722448348999s
Time consumed at iteration 7: 1.388857126235962s
Time consumed at iteration 8: 1.6264557838439941s
Time consumed at iteration 9: 1.456557273864746s
```

# Reference

[[2502.12170] MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections](https://arxiv.org/abs/2502.12170)

[Caiyun-AI/MUDDFormer](https://github.com/Caiyun-AI/MUDDFormer)
