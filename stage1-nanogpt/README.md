# 阶段 ① · 跑通并读懂 nanoGPT

## 目标

拿 nanoGPT 这个现成项目，完整跑一次训练，并把代码读到能解释每一行的程度。

**不是**自己写模型，是**读懂 + 跑通 + 做实验**。

## 结束时的产物

- [ ] 一个能复现的环境（本仓库 `env/requirements.txt` + 系统的 CUDA 记录）
- [ ] 一条自己跑出来的 loss 曲线
- [ ] 一个自己的模型 checkpoint（`checkpoints/`，不入库）
- [ ] 一份「超参为什么在那儿」的笔记（放本目录）
- [ ] 一次故障复盘：故意造 loss 尖刺，再定位修好

## 怎么做

```powershell
# 外部项目放 vendor/，只读参考，不改它的代码
git clone https://github.com/karpathy/nanoGPT.git vendor\nanoGPT

# 跑最小配置
cd vendor\nanoGPT
python train.py config\train_shakespeare_char.py
```

实验脚本写在本目录，**不要改 vendor 里的文件** —— 用 import 或复制配置出来改。

## 本目录放什么

```
stage1-nanogpt/
├── README.md          # 本文件
├── notes/             # 阅读笔记：每个模块在干什么
├── experiments/       # 改超参的脚本、对照结果
└── figures/           # loss 曲线、对照图
```

## 对应任务

#1–#13（见 [docs/工作日志.md](../docs/工作日志.md) 的任务池）

其中最不该跳的是 **#8：故意制造一次 loss 尖刺，再定位修好**。

## 验收（M1）

- 能说出 GPT-2 small 的层数 / 头数 / 维度
- 能解释梯度累积、lr 衰减、weight decay 各自为什么在那儿
- 能回答「只有 8G 显存怎么训 124M」

---

# `model.py` 结构地图（330 行）

仓库已拉到 `vendor/nanoGPT`。这个文件是整个 GPT 的全部定义，**不要从头读到尾**——它不是按阅读顺序写的。

## 一、七个部分

| 行号 | 内容 | 一句话 |
|---|---|---|
| 18–27 | `LayerNorm` | 标准归一化，多一个「可选 bias」参数 |
| 29–76 | `CausalSelfAttention` | **核心**：QKV 投影 → 分头 → 因果注意力 → 合头 |
| 78–92 | `MLP` | 两层全连接 + GELU，中间维度放大 4 倍 |
| 94–106 | `Block` | 一个 transformer 层 = 注意力 + MLP + 两个残差 |
| 108–116 | `GPTConfig` | 六个超参 |
| 118–330 | `GPT` | 总装：embedding + N×Block + 输出层 + 各种工具方法 |

## 二、建议的阅读顺序

分三次，每次一小时左右：

**第 1 次：搞清「长什么样」**
1. `GPTConfig`（108–116）—— 六个数字就定义了一个模型
2. `GPT.__init__`（120–148）—— 组件清单，看清 `wte` `wpe` `h` `ln_f` `lm_head` 各是什么
3. `GPT.forward`（170–193）—— 数据怎么流过去

**第 2 次：搞清「一层是怎么搭的」**
4. `Block`（94–106）—— 只有 3 行，但包含残差和 Pre-LN
5. `MLP`（78–92）—— 4 倍扩张
6. `LayerNorm`（18–27）

**第 3 次：啃核心**
7. `CausalSelfAttention`（29–76）—— 这是全文件最值得反复看的地方

## 三、逐段对照：你学过的东西在这里

| `model.py` | 对应 B1–B5 的哪一节 |
|---|---|
| `k.view(B, T, nh, C//nh).transpose(1, 2)`（57–59） | B1「reshape 和 transpose 不能互换」——就是那个坑 |
| `att / math.sqrt(k.size(-1))`（67） | B1「为什么除以 √d_k」 |
| `masked_fill(bias == 0, -inf)`（68） | causal mask：只能看左边 |
| `F.cross_entropy(logits.view(-1, V), targets.view(-1))`（187） | B5 的交叉熵 |
| `x = x + self.attn(...)`（104–105） | 残差连接 `x + f(x)`：B5 之后第一次见，但结构很简单 |
| `logits[:, -1, :] / temperature`（318） | 温度：softmax 之前除以 T |
| `nn.Embedding(vocab, n_embd)`（127） | B5 里 one-hot + Linear 的等价物 |

## 四、四个容易卡住的地方

**1. `c_attn` 为什么是 `3 * n_embd`（35 行）**

一次线性层同时算出 Q、K、V 三个矩阵，省一次调用。之后 `split(n_embd, dim=2)` 切成三份。

**2. 权重绑定（138 行）**

```python
self.transformer.wte.weight = self.lm_head.weight
```

输入 embedding 和输出投影**共用同一个矩阵**。逻辑上说得通（都是「字符 ↔ 向量」的映射），还能省一大块参数。

**3. 残差投影的特殊初始化（143–145 行）**

```python
std = 0.02 / math.sqrt(2 * n_layer)
```

层数越多，残差累加的值越大，所以按 `1/√(2·n_layer)` 缩小，防止深层模型一开始就数值爆炸。

**4. 推理时只算最后一个位置（190 行）**

训练要算所有位置，推理只需要最后一个 token 的 logits，所以 `x[:, [-1], :]`。这是推理加速的最小白拿。

## 五、几个数字锚点

实例化验证过（CPU）：

| 配置 | 参数量 |
|---|---|
| `GPTConfig()` 默认（= GPT-2 124M） | **123.69M** |
| shakespeare_char（6 层、384 维、词表 65） | **10.67M** |

也就是说 char 模型比 GPT-2 小 **11 倍**，这是它能在 CPU 上跑的原因。
