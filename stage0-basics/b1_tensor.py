"""B1 · 张量基础

跑法（仓库根目录）：
    .venv\\Scripts\\python.exe stage0-basics\\b1_tensor.py

学完这一节，你应该能回答：
  - 看到一个张量的 shape，能说出每个数字代表什么
  - 知道 view / reshape / permute 各在干什么
  - 知道什么时候会发生"广播"，以及它为什么会坑人
  - 知道 @ 和 * 的区别

文件最后有 5 道练习，填完会自动判对错。
"""

import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


# ============================================================ 1. 三件套
section("1. 张量的三件套：shape / dtype / device")

x = torch.randn(2, 3)
print(f"x = {x}")
print(f"shape  (形状) = {tuple(x.shape)}   ← 两个数字分别是什么？")
print(f"dtype  (类型) = {x.dtype}")
print(f"device (设备) = {x.device}")

print("\n读 shape 的心法：从右往左读，最右边是最细的一层")
print("  [2, 3]      = 2 行 3 列")
print("  [2, 3, 4]   = 2 个「3 行 4 列」的矩阵")
print("  [8, 16, 64, 32] = 8 个 batch × 16 个 head × 64 个 token × 32 维")

print("\ndtype 常见的有：")
for dt in [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool]:
    print(f"  {str(dt):<20} 占 {torch.tensor([], dtype=dt).element_size()} 字节")

print("\ndevice：CPU 上是 cpu，GPU 上是 cuda:0。两边不能直接运算，要先 .to(device)")

# 创建一个空张量看这三件套
y = torch.zeros(4, 5, dtype=torch.float16)
print(f"\ntorch.zeros(4, 5, dtype=float16) -> shape={tuple(y.shape)}, dtype={y.dtype}")


# ============================================================ 2. 创建
section("2. 常用创建方式")

print(f"torch.zeros(2, 3)        =\n{torch.zeros(2, 3)}")
print(f"\ntorch.ones(2, 3)         =\n{torch.ones(2, 3)}")
print(f"\ntorch.full((2, 3), 7)    =\n{torch.full((2, 3), 7)}")
print(f"\ntorch.arange(0, 10, 2)   = {torch.arange(0, 10, 2)}")
print(f"torch.linspace(0, 1, 5)  = {torch.linspace(0, 1, 5)}")
print(f"torch.randn(2, 3)        =\n{torch.randn(2, 3)}   ← 标准正态，均值 0 方差 1")
print(f"torch.randint(0, 10, (3,)) = {torch.randint(0, 10, (3,))}")

print("\ntorch.manual_seed(0) 是干什么的？—— 固定随机数，保证两次跑结果一样")
torch.manual_seed(0)
print(f"  设种子后: {torch.randn(3)}")
torch.manual_seed(0)
print(f"  再设一次: {torch.randn(3)}   ← 一样")


# ============================================================ 3. 变形
section("3. 变形：view / reshape / permute / unsqueeze")

a = torch.arange(12)
print(f"a = {a}   shape={tuple(a.shape)}")

b = a.reshape(3, 4)
print(f"\na.reshape(3, 4) =\n{b}")

c = a.reshape(2, 2, 3)
print(f"\na.reshape(2, 2, 3) -> shape={tuple(c.shape)}")

print("\n【view vs reshape】")
print("view   只能用在内存连续时，否则报错；reshape 会自动拷贝，永远能用")
print("新手建议：一律用 reshape，不要纠结 view")

print("\n【-1 是自动推断】")
d = a.reshape(3, -1)
print(f"a.reshape(3, -1) -> shape={tuple(d.shape)}  ← -1 表示「你自己算」")

print("\n【permute：重新排列维度顺序】")
img = torch.randn(3, 32, 32)          # 通道, 高, 宽
print(f"img shape            = {tuple(img.shape)}   ← PyTorch 惯例：通道在前")
print(f"img.permute(1,2,0)   = {tuple(img.permute(1, 2, 0).shape)}   ← 变成 高,宽,通道")
print("注意：permute 不改数据，只改「怎么读」，所以之后运算可能报内存不连续")

print("\n【unsqueeze / squeeze：加/去一个长度为 1 的维度】")
v = torch.tensor([1.0, 2.0, 3.0])
print(f"v           shape={tuple(v.shape)}")
print(f"v.unsqueeze(0) shape={tuple(v.unsqueeze(0).shape)}   ← 变成 1 行 3 列")
print(f"v.unsqueeze(1) shape={tuple(v.unsqueeze(1).shape)}   ← 变成 3 行 1 列")
print("这两个在「凑 batch 维度」时天天用")


# ============================================================ 4. 索引
section("4. 索引与切片")

m = torch.arange(12).reshape(3, 4)
print(f"m =\n{m}")
print(f"\nm[0]        = {m[0]}        ← 第 0 行")
print(f"m[:, 0]     = {m[:, 0]}        ← 第 0 列")
print(f"m[1:, 2:]   =\n{m[1:, 2:]}")
print(f"m[m > 5]    = {m[m > 5]}   ← 布尔索引，取出所有大于 5 的")

idx = torch.tensor([0, 2])
print(f"m[idx]      =\n{m[idx]}   ← 按索引列表取行")


# ============================================================ 5. 广播
section("5. 广播（最常见的坑）")

print("规则一句话：从右往左逐维比较，要么相等，要么其中一个是 1，否则报错。")

x = torch.randn(3, 4)
bias = torch.randn(4)          # shape (4,)
print(f"\nx {tuple(x.shape)} + bias {tuple(bias.shape)}  ->  结果 {tuple((x + bias).shape)}")
print("  bias 被自动复制成 3 行，逐行相加")

x2 = torch.randn(3, 4)
col = torch.randn(3, 1)        # shape (3, 1)
print(f"\nx2 {tuple(x2.shape)} + col {tuple(col.shape)}  ->  结果 {tuple((x2 + col).shape)}")
print("  col 的 1 被自动复制成 4 列")

print("\n【坑】：你以为在做 A，实际在做 B")
a = torch.randn(3, 1)
b = torch.randn(1, 4)
print(f"  a {tuple(a.shape)} + b {tuple(b.shape)} -> {tuple((a + b).shape)}   ← 变成 3×4 矩阵了！")
print("  想逐元素加，两个 shape 必须一致")


# ============================================================ 6. @ 与 *
section("6. @ 和 *：两个完全不同的东西")

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
print(f"a =\n{a}")
print(f"b =\n{b}")
print(f"\na * b （逐元素相乘）=\n{a * b}")
print(f"\na @ b （矩阵乘法）=\n{a @ b}")
print("\n记住：* 是每个位置各乘各的；@ 是线性代数里的矩阵乘。神经网络里几乎都是 @。")

print("\n矩阵乘的 shape 规则：(m, k) @ (k, n) = (m, n)")
v = torch.randn(4)
w = torch.randn(8, 4)
print(f"  v {tuple(v.shape)} @ w.T {tuple(w.T.shape)} -> {tuple((v @ w.T).shape)}   ← 这就是 nn.Linear 干的事")
print("  nn.Linear(in=4, out=8) 内部就是 x @ W.T + b")


# ============================================================ 7. 练习
section("7. 练习（填完自动判分）")

results = []


def check(name: str, fn, hint: str = "") -> None:
    try:
        ok = fn() is True
    except Exception as e:  # noqa: BLE001
        ok = False
        hint = f"出错了：{type(e).__name__}: {e}"
    results.append((name, ok))
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok and hint:
        print(f"      提示：{hint}")


# ---- 练习 1：取第一个 batch ----
# t 的 shape 是 (2, 3, 4)，取出第 0 个 batch，结果应该是 shape (3, 4)
t = torch.arange(24).reshape(2, 3, 4)


def ex1():
    first = t                       # ← 改这一行
    return tuple(first.shape) == (3, 4)


# ---- 练习 2：把 (2,3,4) 压成 (6,4) ----
def ex2():
    flat = t                        # ← 改这一行
    return tuple(flat.shape) == (6, 4)


# ---- 练习 3：把 (3,4) 转置成 (4,3) ----
def ex3():
    m = torch.zeros(3, 4)
    t_m = m                         # ← 改这一行
    return tuple(t_m.shape) == (4, 3)


# ---- 练习 4：用广播给每一行加同一个向量 ----
# x 是 (3, 4)，vec 是 (4,)，让每一行都加上 vec
x = torch.ones(3, 4)
vec = torch.tensor([1.0, 2.0, 3.0, 4.0])


def ex4():
    out = x                         # ← 改这一行
    return tuple(out.shape) == (3, 4) and torch.allclose(
        out[0], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )


# ---- 练习 5：区分 @ 和 * ----
# A 是 (2,3)，B 是 (3,)
#   mul 应为逐元素乘（广播后 shape (2,3)）
#   mat 应为矩阵乘（shape (2,)）
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = torch.tensor([1.0, 10.0, 100.0])


def ex5():
    mul = A                         # ← 改这一行
    mat = A                         # ← 改这一行
    return (
        tuple(mul.shape) == (2, 3)
        and tuple(mat.shape) == (2,)
        and torch.allclose(mul[0], torch.tensor([1.0, 20.0, 300.0]))
        and torch.allclose(mat, torch.tensor([321.0, 654.0]))
    )


check("练习 1：取出第 0 个 batch，shape 应为 (3, 4)", ex1,
      "沿着第 0 维切一片：t[?]")
check("练习 2：reshape 成 (6, 4)", ex2,
      "t.reshape(6, 4)")
check("练习 3：转置成 (4, 3)", ex3,
      "矩阵转置用 .T 或 .transpose(0, 1)")
check("练习 4：广播相加，shape (3,4) 且第 0 行为 [2,3,4,5]", ex4,
      "x + vec 就够，想想为什么不用管行数")
check("练习 5：区分 * 和 @", ex5,
      "MUL 用 *，MAT 用 @；注意 mat 的形状是 (2,) 而不是 (2,1)")

passed = sum(1 for _, ok in results if ok)
print(f"\n结果：{passed}/{len(results)} 通过")
if passed == len(results):
    print("全部通过 —— B1 完成，可以进 B2（autograd）。")
else:
    print("还有没过的：改文件里标了「← 改这一行」的地方，再跑一次。")
    print("实在想不出就看第 3–6 节的例子，答案都在里面。")

print("\n【自问】跑完先别关，想想这三个：")
print("  1. shape (8, 16, 64, 32) 每一步代表什么？")
print("  2. 为什么 reshape 比 view 安全？")
print("  3. `a * b` 和 `a @ b` 什么时候形状相同、结果不同？")
