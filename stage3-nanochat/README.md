# 阶段 ③ · nanochat 全流程

## 目标

跑通一个完整的 ChatGPT 类项目：tokenizer → 预训练 → SFT → RL → 评测 → 推理 → 服务。

补上前两段完全没碰的部分：**后训练、评测、推理服务、成本**。

## 结束时的产物

- [ ] 一个能真的聊天的模型 + Web UI
- [ ] 一次真实训练的账单（约 $48 / 2 小时 8×H100），说得清钱花在哪
- [ ] 评测结果：CORE 分数、`val_bpb`、各任务分数
- [ ] 推理基准：延迟 / 吞吐 / 显存

## 怎么做

```powershell
git clone https://github.com/karpathy/nanochat.git vendor\nanochat

# 先免费跑通：CPU 小规模
cd vendor\nanochat
bash runs/runcpu.sh

# 再花钱：8×H100 跑 speedrun（约 $48 / 2h）
bash runs/speedrun.sh
```

## 本目录放什么

```
stage3-nanochat/
├── README.md          # 本文件
├── notes/             # 读 gpt.py / optim.py / common.py 的笔记
├── configs/           # 改过的配置
├── evals/             # 评测结果
└── cost.md            # 每次租卡的账单记录
```

## 预算

| 项 | 金额 |
|---|---|
| 首次 speedrun | ~$48（spot 可到 ~$15） |
| 零散实验 | $50–150 |
| **合计** | **约 $100–200** |

**前两个阶段应该一直是 $0。** 到这一步之前不租卡。

## 对应任务

#27–#39（见 [docs/工作日志.md](../docs/工作日志.md) 的任务池）

## 验收（M3）

- 有一个能真的聊天的模型（不是补全）
- 能说清 SFT 和 RL 各自改善了什么、大概多少
- 能复述一次训练的成本构成
