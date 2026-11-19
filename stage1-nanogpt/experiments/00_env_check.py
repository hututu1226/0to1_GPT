"""环境自检 + PyTorch 手感自测

跑法（在仓库根目录）：
    .venv\\Scripts\\python.exe stage1-nanogpt\\experiments\\00_env_check.py

这个脚本做四件事：
  1. 报告环境（Python / torch / 设备 / CPU 线程）
  2. 验算：手写的 softmax 和反向传播，跟 torch 对得上吗
  3. 跑一个最小训练循环，看 loss 会不会下降
  4. 测这台机器的算力，估算本地能训多大的模型
"""

import os
import platform
import sys
import time

import torch
import torch.nn as nn

# Windows 控制台默认 GBK，中文会乱码；强制用 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


# ---------------------------------------------------------------- 1. 环境
section("1. 环境")

print(f"Python      : {platform.python_version()}")
print(f"torch       : {torch.__version__}")
print(f"平台        : {platform.system()} {platform.machine()}")
print(f"CPU 核心    : {os.cpu_count()} 逻辑核，torch 使用 {torch.get_num_threads()} 线程")
print(f"CUDA 可用   : {torch.cuda.is_available()}")
print(f"MPS 可用    : {torch.backends.mps.is_available()}")

device = torch.device("cpu")
print(f"实际训练设备: {device}")


# ------------------------------------------------------- 2. 张量与手算对比
section("2. 手算 vs torch")

# 2.1 矩阵乘：用 numpy 独立算一遍，跟 torch 比
a = torch.randn(3, 4)
b = torch.randn(4, 5)
c = a @ b

a_np = a.numpy()
b_np = b.numpy()
c_manual = [[sum(a_np[i, k] * b_np[k, j] for k in range(4)) for j in range(5)] for i in range(3)]
c_manual_t = torch.tensor(c_manual, dtype=torch.float32)
print(f"[matmul] 与手算最大误差: {(c - c_manual_t).abs().max().item():.2e}")

# 2.2 softmax：自己实现，对比官方
x = torch.randn(2, 6)
x_max = x.max(dim=-1, keepdim=True).values          # 减最大值，防止 exp 溢出
exp = torch.exp(x - x_max)
softmax_manual = exp / exp.sum(dim=-1, keepdim=True)
softmax_torch = torch.softmax(x, dim=-1)
print(f"[softmax] 与官方最大误差: {(softmax_manual - softmax_torch).abs().max().item():.2e}")
print(f"[softmax] 每行和         : {softmax_manual.sum(dim=-1).tolist()}")

# 2.3 反向传播：y = x^3 的导数应该是 3x^2
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(f"[autograd] dy/dx @ x=2   : {x.grad.item()}（手算应为 3*2^2 = 12）")


# ---------------------------------------------------------- 3. 最小训练循环
section("3. 最小训练循环：用 MLP 拟合 sin(x)")

torch.manual_seed(0)

# 造数据：x 在 [-pi, pi] 上取 200 个点，y = sin(x)
N = 200
x_data = torch.linspace(-3.14159, 3.14159, N).unsqueeze(1)   # [200, 1]
y_data = torch.sin(x_data)                                   # [200, 1]


def build_model() -> nn.Module:
    """一个小 MLP：1 -> 32 -> 32 -> 1"""
    return nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )


def train(lr: float, steps: int = 800, verbose: bool = True):
    """完整的训练循环：前向 -> 算损失 -> 清梯度 -> 反向 -> 更新"""
    model = build_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for step in range(1, steps + 1):
        pred = model(x_data)          # 1. 前向
        loss = loss_fn(pred, y_data)  # 2. 算损失
        optimizer.zero_grad()         # 3. 清空上一步的梯度
        loss.backward()               # 4. 反向传播：算出每个参数的梯度
        optimizer.step()              # 5. 按梯度更新参数
        history.append(loss.item())

        if verbose and step % 100 == 0:
            print(f"    step {step:>4} | loss {loss.item():>10.6f}")

    with torch.no_grad():
        mae = (model(x_data) - y_data).abs().mean().item()
    return history, mae


print("\n--- lr = 1e-2（收敛快，但中途会来回抖）---")
hist_big, mae_big = train(lr=1e-2)
print(f"    最终平均绝对误差 {mae_big:.4f}，历史最低 loss {min(hist_big):.6f}")

print("\n--- lr = 1e-3（曲线平滑，但更慢）---")
hist_small, mae_small = train(lr=1e-3)
print(f"    最终平均绝对误差 {mae_small:.4f}，历史最低 loss {min(hist_small):.6f}")

print(f"\n看结果：lr=1e-2 最终误差 {mae_big:.4f}，比 lr=1e-3 的 {mae_small:.4f} 更好。")
print("所以 lr=1e-2 并不是【坏的】—— 它只是中途 loss 会跳（0.000003 → 0.002 再回来）。")
print("教训：单点 loss 抖动不代表训练坏了，判断要看趋势和最终指标。")
print("这个直觉在后面看真实训练曲线时非常关键。")


# ------------------------------------------------------------ 4. 算力基准
section("4. 这台机器能跑多快")

n = 1024
a = torch.randn(n, n)
b = torch.randn(n, n)

for _ in range(3):                 # 预热，排除首次编译/分配的开销
    _ = a @ b

iters = 20
t0 = time.perf_counter()
for _ in range(iters):
    _ = a @ b
elapsed = time.perf_counter() - t0

flops = 2 * (n ** 3) * iters       # 一次 n×n 矩阵乘 = 2n³ FLOPs
gflops = flops / elapsed / 1e9
print(f"{n}×{n} 矩阵乘 ×{iters} 次: {elapsed:.2f}s")
print(f"实测算力约: {gflops:.1f} GFLOPS")

# 换算成"训一步 GPT-2 small 要多久"
params = 124e6
batch, seq = 12, 1024
step_flops = 6 * params * batch * seq      # 前向+反向约 6·N·B·T FLOPs
step_seconds = step_flops / (gflops * 1e9)
print(f"\nGPT-2 small 训一步（batch=12, seq=1024）约需: {step_seconds:.1f} 秒")
print(f"（训练 5000 步约需: {step_seconds * 5000 / 3600:.1f} 小时）")
print("如果这个数字超过几小时，说明本地只适合跑 nanoGPT 的字符级小模型。")
