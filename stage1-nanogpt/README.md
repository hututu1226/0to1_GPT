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
