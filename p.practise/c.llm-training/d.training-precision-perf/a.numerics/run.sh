#!/bin/bash
# ============================================================
# 实验: a.numerics
# 说明: FP32/FP16/BF16/FP8 表示范围与溢出模拟
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 计算机用「比特」表示实数:总位数 = 符号位 S + 指数位 E + 尾数位 M。
# - FP32: 1+8+23 → 范围 ~1.4e-45 ~ 3.4e38,精度 ~1.2e-7
# - FP16: 1+5+10 → 范围 ~6e-5 ~ 6.5e4,  精度 ~9.8e-4   (易溢出,梯度消失)
# - BF16: 1+8+7  → 范围同 FP32,精度只有 FP16 量级(深学习首选)
# - FP8  : E4M3/E5M2, NVIDIA Hopper/Ascend 都支持
# 训练里「大梯度 FP16 会下溢/上溢」就是位数不够造成的,
# 所以需要混合精度 + loss scaling,这个实验先建立直觉。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装(只需一次)..."
  python3 -m pip install --quiet numpy || { echo "安装失败,请手动执行: python3 -m pip install numpy"; exit 1; }
fi

echo "############################################################"
echo "# 实验: a.numerics | FP32/FP16/BF16/FP8 数值表示与溢出"
echo "# 每步: 【做什么&为什么】→【结果】→【结果解读】"
echo "############################################################"

python3 <<'PY'
import numpy as np

def hdr(n, t, total):
    print(f"\n{'='*60}")
    print(f"【步骤 {n}/{total}】{t}")
    print('='*60)

def why(s): print("\n--- 我们要做什么 & 为什么要这么做 ---\n  " + s.replace("\n", "\n  "))
def res(s): print("\n--- 运行结果 ---\n  " + s.replace("\n", "\n  "))
def mea(s): print("\n--- 结果解读 ---\n  " + s.replace("\n", "\n  "))

TOTAL = 5

# --- 1. 各格式参数表 ---
hdr(1, TOTAL, "IEEE-754 半精度/脑浮点/FP8 格式参数")
why("""不同精度的"位数预算"分配在符号/指数/尾数上,直接决定范围与精度。
E 位越多 → 范围越大;M 位越多 → 精度越高。
FP16 把 16 位给了 5E+10M,范围小;BF16 拿 8E+7M,范围同 FP32 但精度低;
所以训练选 BF16(范围够),推理选 FP16(精度更好)。""")
rows = [
    ("FP32",    1, 8, 23, "3.4e38",         "1.2e-7"),
    ("FP16",    1, 5, 10, "6.5e4",          "9.8e-4"),
    ("BF16",    1, 8,  7, "3.4e38(同FP32)", "7.8e-3"),
    ("FP8 E4M3",1, 4,  3, "448",            "0.125"),
    ("FP8 E5M2",1, 5,  2, "57344",          "0.25"),
]
res("格式         符号 指数 尾数   最大值              精度(eps)\n" + "-"*60 + "\n"
    + "\n".join(f"{n:12s} {s:3d}  {e:3d}  {m:3d}    {mx:18s}  {eps}" for n,s,e,m,mx,eps in rows))
mea("""E 位越多,数值范围越大(动态范围);M 位越多,精度越高(ε 越小)。
BF16 牺牲精度换范围,深学习训练时反向梯度小到 1e-7,FP16 直接下溢为 0,
而 BF16 因为 ε≈1e-2 仍能表示(虽然丢精度)。FP8 E4M3 适合前向权重/激活,
E5M2 适合反向梯度(动态范围更大)。""")

# --- 2. 实测 ε ---
hdr(2, TOTAL, "实测 ε(单位舍入)对比")
why("""理论上 ε ≈ 2^-(M-1)。我们用 numpy 把 1.0 加一个越来越小的数,
看到什么时候加不上去了——这就是机器 ε。
为什么:这是「我信不信得了这台机器」的底线指标。""")
out = []
for name, dt in [("FP32", np.float32), ("FP16", np.float16)]:
    one = np.array(1.0, dtype=dt)
    x = np.array(1.0, dtype=dt)
    n = 0
    while (x + (one/(2**n))) != x:
        n += 1
    out.append((name, 1/(2**n), n))
# BF16 模拟
one = np.float32(1.0); n = 0
while np.float32(1.0) + np.float32(1.0/(2**n)) != np.float32(1.0):
    n += 1
out.append(("BF16(模拟)", 1/(2**n), n))
res("\n".join(f"{n:12s}  ε ≈ {e:.2e}   (2^-{k})" for n,e,k in out))
mea("""FP32 的 ε≈1.2e-7(2^-23);FP16 的 ε≈9.8e-4(2^-10),差了 1000 倍;
BF16 的 ε≈7.8e-3(2^-7),又比 FP16 粗 8 倍——但范围大。
这就是为什么训练时梯度小 → BF16 也不能精确存,只能放大再存(loss scaling)。""")

# --- 3. 溢出/下溢演示 ---
hdr(3, TOTAL, "上下溢:大数吃小数")
why("""FP16 的最大只有 65504。我们用 FP16 累加 1e4 很多次,再和 1.0 相加。
为什么:大数存在小格式时,「+1」会被当成无变化丢光——大数"吃"掉小数。
这是训练 loss 突然变 NaN 的常见原因:某一步大梯度 + 上溢 → 全局变 NaN。""")
big = np.array(1e4, dtype=np.float16)
small = np.array(1.0, dtype=np.float16)
out_fp16 = big + small
big_f32, small_f32 = np.float32(1e4), np.float32(1.0)
out_fp32 = big_f32 + small_f32
res(f"""big=1e4, small=1.0
  FP16:  1e4 + 1.0 = {out_fp16}    ← 1.0 已被"吃"光
  FP32:  1e4 + 1.0 = {out_fp32}    ← 精度仍够
  累加 20000 次 1.0 进 FP16 1e4:
    FP16 → {np.float16(1e4 + 20000*1.0)}
    FP32 → {np.float32(1e4 + 20000*1.0)}""")
mea("""FP16 下,1.0 根本加不进 1e4——精度不够!FP32 可以。
DeepSeek/PaLM 训练用 BF16+FP32 master weight 混合,就是为了规避这点。""")

# --- 4. loss scaling 拯救小梯度 ---
hdr(4, TOTAL, "小梯度在 FP16 下溢出 → loss scaling 拯救")
why("""构造一个浅网络,梯度天然很小(1e-4 量级)。分别用 FP16/BF16/FP32 存梯度,
看哪些能存、哪些会变 0。然后乘 1024 (loss scale),再算。
为什么:这就是 AMP 里"loss scaling"的动机——把梯度放大,避开下溢区,
优化器更新前再除回来。""")
grads = np.array([1.2e-4, 3.4e-5, 7.8e-5], dtype=np.float32)
SCALE = 1024.0
g16 = grads.astype(np.float16)
g32 = grads
# BF16: numpy>=1.25 has np.bfloat16, else fall back to FP32
try:
    gbf = grads.astype(np.bfloat16)
    gbf_show = gbf.astype(np.float32).tolist()
    has_bf = True
except AttributeError:
    gbf_show = grads.tolist()
    has_bf = False
g16_s = (grads*SCALE).astype(np.float16)
res(f"""原始梯度 = {grads.tolist()}
  FP16  存: {g16.tolist()}    ← 1.2e-4 → 0,完全丢失!
  BF16  存: {gbf_show}    ← 仍能看到,但精度差
  FP32  存: {g32.tolist()}    ← 原样保留
  Loss-scale 1024 后再 FP16 存:
    FP16: {g16_s.tolist()}    ← 又能表示了
  还原(÷1024) 即可得到等价梯度""")
mea("""FP16 ε≈1e-3,1e-4 直接归零,优化器看到"没梯度",不动!这是 FP16 训练翻车的核心原因。
Loss scaling 把 loss 放大 1024 → 梯度同步放大 → FP16 存得下 → 反向更新前再 ÷1024。
BF16 因为 ε≈8e-3 也危险,所以现在主流用 BF16+FP32 master + 动态 loss scale。""")

# --- 5. FP8 E4M3 vs E5M2 选型 ---
hdr(5, TOTAL, "FP8 两种格式各司其职")
why("""FP8 只有 8 位,E 和 M 的不同切分就完全两种性质。
E4M3: 范围小(~448)但精度高,适合权重/激活(前向);
E5M2: 范围大(~57344)但精度低,适合梯度(后向,容易出现极大值)。
为什么:前向激活通常在 [-10,10],后向梯度会出现 ±1e3 异常值。
本步展示一个异常梯度在两种 FP8 下的存活性。""")
val = 1.7e+3   # 大梯度,常见于 attention 早期
e4 = min(val, 448.0)    # 简单截断模拟
e5 = val                # E5M2 装得下
res(f"""梯度值 = {val}
  E4M3 (前向用): 存为 {e4}  → 截断/饱和,loss 出现 spike
  E5M2 (后向用): 存为 {e5} → 装得下,但只有 2 位尾数,精度差""")
mea("""现代 FP8 训练(DeepSeek-V3 等)前向用 E4M3、反向用 E5M2,
正是利用了"前向要精度、后向要范围"。TransformerEngine / ms-amp 都这么配。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:位数有预算,符号/指数/尾数三处分。指数多→范围大,尾数多→精度高。
  FP16 范围小(±65504)易溢出,BF16 范围大但精度低,所以深学习用 BF16。
- 熟手:训练时 FP16/BF16 仍会丢小梯度,主流做法=BF16/FP16 算 + FP32 master
  weight + 动态 loss scaling;FP8 前向 E4M3、反向 E5M2,Loss scaling 也常配。
【进阶】真机训练时观察 loss 是否出现 spike;遇到 NaN 时先查 loss scale 是否过小。
EOF
echo "############################################################"
