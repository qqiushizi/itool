#!/bin/bash
# ============================================================
# 实验: a.linear-algebra
# 说明: 向量/矩阵/张量、秩、SVD 直观理解(numpy 模拟分解与降维)
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ------------------------------------------------------------
# 配套体验代码: demo.py  (可调参: python3 demo.py --k 1 --seed 7)
# ============================================================
# 【第一性原理】
# 线性代数是 AI 的"语言"。神经网络本质就是一堆矩阵乘法 + 非线性。
# 本实验分 5 步,每步先讲"做什么、为什么",再展示结果,再解读"结果说明什么"。
# ============================================================
set -euo pipefail

# ---- 依赖自检:缺 numpy 则自动安装 ----
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装(只需一次)..."
  python3 -m pip install --quiet numpy || { echo "安装失败,请手动执行: python3 -m pip install numpy"; exit 1; }
fi

echo "############################################################"
echo "# 实验 1: 向量 / 矩阵 / 张量 / 秩 / SVD"
echo "# 线性代数是 AI 的数学语言,本实验边跑边讲,每步分三段:"
echo "#   【做什么&为什么】→【结果】→【结果解读】"
echo "############################################################"

python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def step(n, total, title):
    print(f"\n{'='*60}")
    print(f"【步骤 {n}/{total}】{title}")
    print('='*60)

def explain_why(text):
    print("\n--- 我们要做什么 & 为什么要这么做 ---")
    for line in text.strip().splitlines():
        print("  " + line)

def show_result(text):
    print("\n--- 运行结果 ---")
    for line in text.strip().splitlines():
        print("  " + line)

def read_meaning(text):
    print("\n--- 结果解读(这说明了什么)---")
    for line in text.strip().splitlines():
        print("  " + line)

TOTAL = 5

# ============================================================
step(1, TOTAL, "向量:点积与夹角")
explain_why("""
取两个向量 a、b,算点积 a·b 和它们的夹角 θ。
为什么:点积衡量两个向量"方向有多一致"。Transformer 的注意力
机制里 Q·K(查询·键)本质就是点积——点积越大越"相关"。
所以理解点积 = 理解注意力打分的数学根基。""")
a = np.array([3.0, 4.0]); b = np.array([4.0, 0.0])
dot = np.dot(a, b)
na, nb = np.linalg.norm(a), np.linalg.norm(b)
cos_theta = dot / (na * nb)
theta = np.degrees(np.arccos(cos_theta))
show_result(f"""
a = {a.tolist()},  b = {b.tolist()}
点积 a·b = {dot}
长度 |a| = {na:.3f},  |b| = {nb:.3f}
cosθ = {cos_theta:.3f}  →  夹角 θ = {theta:.1f}°""")
read_meaning(f"""
夹角 {theta:.0f}° 介于 0°(完全同向)和 90°(垂直)之间,说明 a 和 b
方向"有点像但不完全一样"。
记住三条:点积为正→方向基本同向;为负→反向;为0→垂直(无关)。
→ 注意力分数 = Q 与 K 的点积,越大代表越相关,这就是打分原理。""")

# ============================================================
step(2, TOTAL, "矩阵 = 线性变换:看基向量被搬到哪")
explain_why("""
对一个矩阵 M,把基向量 i=(1,0)、j=(0,1) 乘上去,看它们变到哪。
为什么:矩阵就是"搬运规则"。看它把最简单的两个向量搬到哪里,
就懂了这个矩阵在做什么变换(旋转?拉伸?压缩?)。
神经网络每一层权重都是这样的"搬运矩阵"。""")
theta = np.radians(30)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
S = np.diag([2.0, 0.5])
unit = np.array([[1,0],[0,1]]).astype(float)
Rout = R.dot(unit); Sout = S.dot(unit)
show_result(f"""
旋转矩阵 R(30°) 作用于基向量 i,j:
  i=(1,0) → {Rout[0].round(3).tolist()}
  j=(0,1) → {Rout[1].round(3).tolist()}
缩放矩阵 S=diag(2, 0.5) 作用于基向量 i,j:
  i=(1,0) → {Sout[0].round(3).tolist()}   (x 被拉长到 2)
  j=(0,1) → {Sout[1].round(3).tolist()}   (y 被压缩到 0.5)""")
read_meaning("""
R 把 i、j 整体旋转了 30°,长度不变——所以 R 是"纯旋转"。
S 把 x 方向拉长 2 倍、y 方向压到一半——所以 S 是"拉伸/压缩"。
任意矩阵 = 旋转 + 拉伸的组合。理解这点就理解了"神经网络怎么变换数据"。""")

# ============================================================
step(3, TOTAL, '秩:矩阵里有多少独立方向')
explain_why("""
比较两个矩阵:一个两列不共线,一个第二列是第一列的倍数。算它们的秩。
为什么:秩 = 矩阵里"真正独立、不冗余"的方向数。秩低 = 有冗余 =
信息可以被更少维度表达。LoRA 用低秩矩阵近似权重更新,正是利用这点。""")
M_full = np.array([[1,2],[3,4]])
M_low  = np.array([[1,2],[2,4]])
show_result(f"""
满秩矩阵:            降秩矩阵:
  {M_full[0].tolist()}        {M_low[0].tolist()}
  {M_full[1].tolist()}        {M_low[1].tolist()}   (第2行 = 2×第1行)
秩 = {np.linalg.matrix_rank(M_full)}                  秩 = {np.linalg.matrix_rank(M_low)}  (有冗余)""")
read_meaning("""
满秩=2:两列各自独立,信息"满"。降秩=1:第二列只是第一列的放大,
没有新信息,可以用 1 个方向就表达。
→ LoRA 的巧思:大模型权重更新 ΔW 是高维的,但它假设 ΔW 是"低秩"的,
  用两个小矩阵 A·B 来近似,参数量骤降,这就是低秩的威力。""")

# ============================================================
step(4, TOTAL, "SVD:把任意矩阵拆成 旋转-拉伸-旋转")
explain_why("""
对一个矩阵做 SVD 分解,得到 U、S、V,再只用最大的 1 个奇异值重建它。
为什么:SVD 证明任意矩阵都能拆成"旋转→沿轴拉伸→旋转"三步,
拉伸强度就是奇异值 S(从大到小排)。丢掉小奇异值 = 丢次要信息 = 降维压缩。
这是 PCA、推荐系统、模型压缩共同的数学根基。""")
A = np.array([[3,2,2],[2,3,-2]])
U, S, Vt = np.linalg.svd(A, full_matrices=False)
recon = U @ np.diag(S) @ Vt
k = 1
A_low = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
keep = np.sum(S[:k]**2) / np.sum(S**2)
show_result(f"""
原矩阵 A =                奇异值 S = {S.round(3).tolist()}
  {A[0].tolist()}            (从大到小 = 各方向信息强度)
  {A[1].tolist()}
重构(用全部奇异值)误差 = {np.max(np.abs(recon-A)):.2e}  (≈0,完美还原)
只保留前 {k} 个奇异值的低秩近似:
  {A_low[0].round(3).tolist()}
  {A_low[1].round(3).tolist()}
能量保留率 = {keep*100:.1f}%  (保留的奇异值平方和占比)""")
read_meaning(f"""
用全部奇异值能完美还原原矩阵(误差≈0)。
只留 1 个最大的奇异值,虽然矩阵变了,但保留了 {keep*100:.0f}% 的"能量"(主要信息)。
→ 奇异值越大越是"主成分",越小越是"噪声/细节"。压缩 = 留大丢小。
→ 试一试:运行 demo.py 用 --k 1 或 --k 2,看保留不同数量时的近似效果。""")

# ============================================================
step(5, TOTAL, "张量:向高维推广(AI 数据的通用形态)")
explain_why("""
构造一个 3 阶张量(2×3×4),看它的形状和切片。
为什么:向量是 1 阶张量,矩阵是 2 阶,更高维统称张量。AI 里所有数据
都是张量:图像=[批,高,宽,通道]、文本=[批,序列,词向量]。
shape(形状)是张量最重要的属性,决定数据怎么流动。""")
T = np.arange(2*3*4).reshape(2,3,4)
show_result(f"""
3 阶张量 shape = {T.shape}  (可看作 {T.shape[0]} 个 {T.shape[1]}×{T.shape[2]} 矩阵叠起来)
共 {T.size} 个元素
切片 T[0] = 第 0 个矩阵:
  {T[0][0].tolist()}
  {T[0][1].tolist()}
  {T[0][2].tolist()}""")
read_meaning("""
shape=(2,3,4):2 个"层",每层 3 行 4 列。改一下理解方式:
图像 [批,高,宽,通道] —— 比如 [32, 224, 224, 3] = 32 张 224×224 的 RGB 图。
文本 [批,序列,词向量] —— 比如 [4, 128, 768] = 4 句话、每句 128 词、每词 768 维。
→ 调试模型时,最常出错的就是 shape 不匹配。看懂 shape = 看懂数据怎么流。""")
PY

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:向量是有方向的数,矩阵是变换规则(看基向量被搬到哪),张量是多维数组;
  秩看"有多少独立信息",SVD 把矩阵拆成旋转-拉伸-旋转、靠丢小奇异值来压缩。
- 熟手:AI 的性能与显存几乎都在和 shape/秩打交道;SVD 低秩是 LoRA、PCA、
  模型压缩的数学根基;奇异值衰减快慢决定"可压缩到什么程度"。

【动手体验】本目录还有 demo.py,可以自己调参数玩:
  python3 demo.py              # 默认演示
  python3 demo.py --k 1        # 只保留 1 个奇异值,看近似效果
  python3 demo.py --k 2        # 保留 2 个,对比精度
  python3 demo.py --seed 7     # 换随机矩阵
  python3 demo.py --rows 3 --cols 5   # 自定义矩阵大小
EOF
echo "############################################################"
