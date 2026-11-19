# 阶段 ② · 从零手写 GPT

## 目标

不看 nanoGPT，从空文件写出自己的 GPT：attention、transformer block、训练循环、BPE tokenizer、KV cache。

**这是整个仓库的主体。**

## 结束时的产物

- [ ] 一套 500–1500 行的自己的实现
- [ ] 一个自己训练出来的、能生成可读英文的小模型
- [ ] 一个手写的 BPE tokenizer
- [ ] 一组消融结果（RoPE / Pre-LN / SwiGLU 各去掉会怎样）
- [ ] 通过断电测试

## 建议的文件划分

```
stage2-from-scratch/
├── model.py         # attention / block / GPT
├── train.py         # 训练循环
├── tokenizer.py     # 手写 BPE
├── sample.py        # 采样 + KV cache 推理
├── configs/         # 超参配置
└── tests/           # 数值对齐测试（对着 PyTorch 官方实现比）
```

## 规则

1. **不许复制 nanoGPT 的代码**。可以查资料、可以看论文，但文件是空着开始敲的。
2. **每一步都要能数值对齐**。写完 attention，拿 `nn.MultiheadAttention` 比输出，误差 < 1e-5 才算对。
3. **断电测试**：合上所有资料，2–3 小时从空文件写出能训练、能采样的 GPT。

## 对应任务

#14–#26（见 [docs/工作日志.md](../docs/工作日志.md) 的任务池）

**#18 手写 BPE 最容易被跳过，别跳。** 它是阶段 ③ 里最容易看不懂的地方的根源。

## 验收（M2）

- 断电测试通过
- 能手写 BPE（不查资料）
- 能说出「我的实现为什么比 `nn.MultiheadAttention` 慢」
