# PyTorch 入门

> 阶段 ⓪ 的笔记。内容为 2026-11-24 整理的手写笔记，正文原样保留，仅标题层级统一降了一级。
> 原始文件：`手写笔记_线性回归到Softmax.md`

## PyTorch 基础操作与从线性回归到 Softmax 的推导笔记

> 整理时间：2026-11-24 16:08
>
> 内容分两部分：前四节为 PyTorch 的常用操作（张量、梯度、模型、训练循环），后四节为手写笔记整理的推导过程（线性回归训练流程、训练术语、Softmax 与交叉熵、前向形状）。

### 一、张量的形状操作

#### 1. 形状查询

```python
x = torch.randn(2, 3)
x.shape          # torch.Size([2, 3])
tuple(x.shape)   # (2, 3)
x.shape[0]       # 2
```

`x.shape` 返回 `torch.Size`，它是 `tuple` 的子类，可直接按元组使用（索引、解包、与元组比较）。`tuple(x.shape)` 得到普通元组。

#### 2. 张量创建与随机种子

```python
torch.randn(2, 3)        # 标准正态分布随机张量，形状 2×3
torch.manual_seed(0)     # 固定随机种子
```

`torch.randn` 生成均值为 0、标准差为 1 的正态分布随机数，神经网络权重初始化常用该函数。`torch.manual_seed` 固定随机数序列，使多次运行的初始化结果一致，便于复现。

#### 3. 形状变换

```python
x.reshape(B, H, T, D)    # 拆成 batch / head / token / dimension 四维
x.reshape(3, -1)         # 第二维由元素总数自动推算
```

`reshape` 按行优先顺序重新解释元素排布，要求变换前后元素总数一致。参数写 `-1` 时该维度由元素总数与其余维度自动推算，只允许出现一个 `-1`。

与 `reshape` 相关的 `view` 只能作用于内存连续（contiguous）的张量；`reshape` 在内存不连续时内部会先复制，适用范围更宽。

#### 4. 维度顺序变换

```python
x.permute(1, 2, 0)       # 任意重排全部维度
x.transpose(0, 1)        # 只交换两个维度
```

`permute` 的参数是新维度顺序，上例把原来的第 0、1、2 维依次放到第 1、2、0 位。`transpose` 是 `permute` 在「只交换两维」情形下的特例。

两者均只改变索引方式，不改变底层数据排布，返回的是视图。变换后的张量通常不连续，若后续要用 `view`，需先调用 `.contiguous()`。

#### 5. 增加与删除长度为 1 的维度

```python
x.unsqueeze(0)           # 在第 0 维前插入一个长度为 1 的维度
x.squeeze(0)             # 删除第 0 维（要求该维长度为 1）
```

若 `x` 的形状为 `(3,)`，`x.unsqueeze(0).shape` 为 `(1, 3)`。常见用途是给单个样本补上 batch 维。

#### 6. 矩阵乘法与逐元素乘法

```python
A @ B                    # 矩阵乘法，等价于 torch.matmul(A, B)
A * B                    # 逐元素相乘
```

`@` 要求 `A` 的最后一维与 `B` 的倒数第二维相等，不符合则报维度不匹配。`*` 要求两形状相同或可广播，得到逐元素乘积。

两种运算语义完全不同：线性层内部用的是矩阵乘法 `W @ x`，注意力权重的加权求和也用矩阵乘法，而掩码（mask）、缩放系数这类操作使用逐元素乘法。

### 二、梯度的计算

#### 1. 标记需要求导的张量

```python
x = torch.tensor(2.0, requires_grad=True)
y = 2 * x
y.backward()
x.grad                   # tensor(2.)
```

`requires_grad=True` 使 PyTorch 记录对该张量的后续所有运算，逐步构建计算图。未标记的张量不参与记录，也不会得到梯度。

#### 2. 反向传播与梯度读取

`y.backward()` 从标量输出出发沿计算图反向应用链式法则，把每个叶节点的梯度写入其 `.grad` 属性。上例中 $y = 2x$，故 $x.\text{grad} = 2$。

`.grad` 在首次反向传播之前为 `None`，之后为张量。

#### 3. 梯度清零

```python
w.grad.zero_()           # 原地清零某个参数的梯度
optimizer.zero_grad()    # 清零优化器管理的全部参数梯度
```

清零是对 `.grad` 这个张量调用原地方法 `zero_`（PyTorch 约定：方法名以下划线结尾表示原地修改）。

不清零会导致梯度累加：`.grad` 每次反向传播都是**累加**而非覆盖写入。这一设计用于需要多次反向传播再统一更新的场景，代价是常规训练循环中每轮必须手动清零，否则本次梯度会叠加历史梯度。

#### 4. 关闭梯度计算

```python
with torch.no_grad():
    pred = model(x)
```

该上下文管理器内的运算不构建计算图，不保存中间结果。推理阶段使用可显著降低显存占用并提高速度。评估与验证代码应放在该上下文内。

### 三、模型的构建

#### 1. 顺序容器

```python
model = nn.Sequential(
    nn.Linear(4, 2),
    nn.ReLU(),
)
```

`nn.Sequential` 按传入顺序串联各层，前一层输出作为后一层输入。

`nn.Linear(in_features, out_features)` 的权重形状为 `(out_features, in_features)`，输入形状为 `(batch, in_features)`，输出形状为 `(batch, out_features)`。

#### 2. 训练状态与推理状态

```python
model.train()            # 切换到训练状态
model.eval()             # 切换到推理状态
```

这两个方法只切换模块内部的 `training` 标志，不修改任何参数数值。受影响的是 Dropout（训练时按概率丢弃、推理时全部保留）和 BatchNorm（训练时用当前批统计量、推理时用滑动平均统计量）等行为随状态变化的层。忘记切换会导致推理结果不稳定。

#### 3. 参数遍历与冻结

```python
for p in model.fc1.parameters():
    p.requires_grad = False          # 冻结该层参数
```

`parameters()` 递归返回该模块及其子模块的全部可训练参数。把 `requires_grad` 置为 `False` 即冻结，反向传播时不再为该参数计算梯度，优化器也不会更新它。等价写法为 `p.requires_grad_(False)`。

微调预训练模型时通常冻结主干、只训练新增的任务头。

### 四、训练循环

#### 1. 五个步骤

```python
pred = model(x)                  # 1. 前向：算预测
loss = loss_fn(pred, y)          # 2. 算损失
optimizer.zero_grad()            # 3. 清空旧梯度
loss.backward()                  # 4. 反向：算新梯度
optimizer.step()                 # 5. 按梯度更新参数
```

| 步骤 | 作用 | 需要自行定义的部分 |
| --- | --- | --- |
| 前向 `model(x)` | 沿网络计算输出 | 模型结构 `model` |
| 损失 `loss_fn(pred, y)` | 度量预测与标签的差距 | 损失函数 `loss_fn` |
| 清零 `zero_grad()` | 清除上一轮累积的梯度 | 无（优化器自带） |
| 反向 `backward()` | 沿计算图求各参数梯度 | 无（自动微分） |
| 更新 `step()` | 按梯度与学习率修改参数 | 优化器类型与学习率 |

顺序不可调换：`zero_grad` 必须在 `backward` 之前，否则清零会抹掉本轮刚算出的梯度；`step` 必须在 `backward` 之后，因为它要读取 `.grad`。

#### 2. 嵌套结构

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        # 在此评估
        pass
```

外层循环遍历轮次（epoch），内层循环遍历批次（batch）。每个批次执行一次完整的五步更新，因此一轮内的参数更新次数等于批次数量。

### 五、线性回归的训练流程

#### 1. 计算预测值

目标函数（示例）：

$$
y = 2x + 4
$$

模型预测值：

$$
\text{pred} = w \cdot x + b
$$

#### 2. 计算损失

$$
L = \text{loss} = (y - \text{pred})^2
$$

该形式即 MSE（Mean Squared Error，均方误差）。

#### 3. 计算梯度

按链式法则展开：

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \text{pred}} \cdot \frac{\partial \text{pred}}{\partial w} = 2(\text{pred} - y) \cdot x
$$

#### 4. 更新参数

$$
w = w - \text{lr} \cdot \frac{\partial L}{\partial w}
$$

$$
b = b - \text{lr} \cdot \frac{\partial L}{\partial b}
$$

其中 lr 为学习率（learning rate）。

#### 5. 循环

用更新后的参数回到第 1 步重新计算，反复迭代至损失收敛。

### 六、三个训练术语的区分

| 术语 | 含义 |
| --- | --- |
| step | 用一批数据完成一次参数更新 |
| batch | 一次参数更新所使用的那批数据 |
| epoch | 遍历完所有数据算作一轮；一轮由若干个 step 组成 |

### 七、由 MSE 扩展到 Softmax 与交叉熵

#### 1. 损失函数的另一种选择

计算完损失后，除 MSE 之外还可以采用 softmax 配合交叉熵的形式，适用于分类任务。

#### 2. Softmax

$$
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

其中 $z_i$ 为经过线性层计算得到的 logits。

#### 3. 交叉熵损失

$$
L = -\log\left(P_{\text{正确分类}}\right)
$$

#### 4. 带温度的 Softmax

$$
P_i = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}
$$

其中 $\tau$ 为温度参数（temperature），用于调节输出分布的平缓程度。

### 八、两层网络的前向形状推导

线性层与激活的堆叠：

$$
f_{c1} = W_1 x + b_1
$$

$$
\text{ReLU}(x)
$$

$$
f_{c2} = W_2 x + b_2
$$

各步张量形状：

| 环节 | 权重形状 | 输入形状 | 输出形状 |
| --- | --- | --- | --- |
| 第一层 $f_{c1}$ | $4 \times 29$ | $29 \times 1$ | $4 \times 1$ |
| 第二层 $f_{c2}$ | $29 \times 4$ | $4 \times 1$ | $29 \times 1$ |

经第二层后得到 logits，形状为 $29 \times 1$。
