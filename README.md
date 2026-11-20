# 0to1_GPT

> 从零构建一个 GPT：**读懂 → 手写 → 交付**。
> 这个仓库既是代码，也是一份公开的工作日志 —— 记录每一步做了什么、卡在哪、学到了什么。

## 为什么有这个仓库

市面上"从零实现 GPT"的项目不少，但大多只覆盖一环：

- [nanoGPT](https://github.com/karpathy/nanoGPT) 只做预训练
- [nanochat](https://github.com/karpathy/nanochat) 做了完整链路，但架构是别人写好的

这个仓库的目标是把三段连起来，并且**每一段都留下证据**（代码、曲线、复盘）。

不追求做出能打的模型，追求把整条链路从黑盒变成白盒。

## 三个阶段

| 阶段 | 主题 | 核心产出 | 任务 |
|---|---|---|---|
| ① | 跑通并读懂 nanoGPT | 自己的 loss 曲线、checkpoint、超参笔记、一次故障复盘 | #1–#13 |
| ② | 从零手写 GPT | 500–1500 行自己的实现、手写 BPE、KV cache、消融实验 | #14–#26 |
| ③ | nanochat 全流程 | 一个能真的聊天的模型、$48 账单、评测与推理基准 | #27–#39 |

一句话：**① 会开，② 会造，③ 会交付。**

> **前置**：另有**阶段 ⓪ PyTorch 基础**（B1–B5，约 20h），在 [stage0-basics/](stage0-basics/)。
> 先做完它，进阶段 ① 才读得懂 `model.py`。

## 进度

| 里程碑 | 状态 |
|---|---|
| M1 · 跑通并理解 nanoGPT | ⬜ 未开始 |
| M2 · 断电测试通过 | ⬜ 未开始 |
| M3 · 有能真的聊天的模型 | ⬜ 未开始 |

| 统计 | 值 |
|---|---|
| 任务完成 | 0 / 44 |
| 累计投入 | 1 h |
| 累计花费 | $0 |

👉 详细记录：**[docs/工作日志.md](docs/工作日志.md)** —— 每动手一次记一笔
👉 任务池与顺序建议：**[docs/路线参考.md](docs/路线参考.md)**

## 目录结构

```
0to1_GPT/
├── docs/
│   ├── 工作日志.md          # 主记录：每动手一次记一笔
│   └── 路线参考.md          # 39 个任务 + 依赖顺序建议
├── notes/
│   ├── README.md            # 笔记索引 + 流程
│   ├── TEMPLATE.md          # 笔记模板（可选）
│   ├── _draft.md            # 随手写的草稿位
│   └── pytorch入门.md       # 阶段 ⓪ 的全部笔记（一个阶段一份）
├── stage0-basics/           # ⓪ PyTorch 基础（Jupyter notebook）
├── stage1-nanogpt/          # ① 对 nanoGPT 的阅读、实验、复盘
├── stage2-from-scratch/     # ② 自己手写的实现（本仓库主体）
├── stage3-nanochat/         # ③ nanochat 的配置、评测、成本记录
├── env/
│   └── requirements.txt
├── vendor/                  # 外部仓库克隆处（不入库）
│   ├── nanoGPT/
│   └── nanochat/
├── data/                    # 数据集（不入库）
├── checkpoints/             # 模型权重（不入库）
└── runs/                    # 训练日志与曲线（不入库）
```

## 快速开始

```powershell
git clone https://github.com/hututu1226/0to1_GPT.git
cd 0to1_GPT

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r env\requirements.txt
pip install jupyterlab ipykernel
python -m ipykernel install --user --name 0to1-gpt --display-name "0to1_GPT (Python 3.12)"

# 拉外部项目（放 vendor/，不入库）
git clone https://github.com/karpathy/nanoGPT.git  vendor\nanoGPT
git clone https://github.com/karpathy/nanochat.git vendor\nanochat
```

打开 notebook：用 **VS Code** 打开这个文件夹，点开 `stage0-basics\b1_tensor.ipynb`，右上角选 kernel → `0to1_GPT (Python 3.12)`。

> 仓库自带 `.vscode/settings.json`，默认解释器已指向 `.venv`，一般不用手选。

> **网络**：国内访问 GitHub 常被重置，需要走代理。若本机代理端口是 7890：
>
> ```powershell
> git config --global http.proxy http://127.0.0.1:7890
> ```
>
> 用完后可以用 `git config --global --unset http.proxy` 取消。

## 常见问题

**notebook 报 `requires the ipykernel package`**

说明 VS Code 选的是 uv 托管的基础 Python（`...\uv\python\cpython-3.12-...\python.exe`），不是项目 venv。

- **不要**执行提示里那条 `pip install ipykernel --user`，那会装进全局解释器
- 修复一：`Ctrl+Shift+P` → **Python: Select Interpreter** → 选 `Python 3.12.14 ('.venv': venv)`
- 修复二：notebook 右上角点内核名 → **Select Another Kernel** → **Python Environments** → 选同一个
- 确认：右上角应显示带 `.venv` 标识的 kernel 名

> 用 VS Code 打开的必须是 `0to1_GPT` 这一层，不是它的父目录，否则 `.vscode/settings.json` 不生效。

**移动或重命名仓库文件夹后，`.venv` 会失效**

uv 生成的 `.venv\Scripts\jupyter.exe` 等启动器里写死了绝对路径。表现是 `jupyter` 命令报 `uv trampoline failed to canonicalize script path`。

重建步骤（**先确认没有 kernel 在跑**）：

```powershell
Stop-Process -Id <占用 .venv 的 python 进程> -Force
Remove-Item .venv -Recurse -Force
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe numpy tiktoken matplotlib tqdm jupyterlab ipykernel
```

## 每个阶段怎么用

**① `stage1-nanogpt/`**
nanoGPT 的源码放在 `vendor/nanoGPT`（只读参考，不改）。自己的实验脚本、改过的超参、曲线图、故障复盘放这里。

**② `stage2-from-scratch/`**
本仓库的主体。不看 nanoGPT，从空文件写出自己的实现。建议的文件划分：

```
model.py       # attention / block / GPT
train.py       # 训练循环
tokenizer.py   # 手写 BPE
sample.py      # 采样与 KV cache 推理
```

**③ `stage3-nanochat/`**
nanochat 源码在 `vendor/nanochat`。这里放：改过的配置、评测结果、成本记录、和自研实现的对照笔记。

## 记录约定

1. 每次动手，在 [docs/工作日志.md](docs/工作日志.md) **顶部**加一条（模板见 [docs/日志模板.md](docs/日志模板.md)）
2. 每学完一节，在 [notes/](notes/) 里留一份笔记（模板见 [notes/TEMPLATE.md](notes/TEMPLATE.md)）
2. 顺手更新 README 与工作日志里的「进度」表
3. 产出放对应 `stage*/` 目录；数据集、checkpoint、`runs/` 不进 git
4. 卡住超过 2 小时就换任务，把现象记进工作日志的「卡住」

## 参考资源

| 资源 | 用途 |
|---|---|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | 阶段 ① 主体 |
| [nanochat](https://github.com/karpathy/nanochat) | 阶段 ③ 主体 |
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 阶段 ② 参考书 |
| [Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 视频补充 |
| [llm.c](https://github.com/karpathy/llm.c) | 想往下挖 C / CUDA |
| [CS336](https://stanford-cs336.github.io/) | 想系统化补课 |

## 说明

- 个人学习仓库，不追求 SOTA，只追求「能复现、能解释」
- 外部项目通过 `vendor/` 引入，不 vendored 进本仓库
- 训练产物、数据集、模型权重均不入库
