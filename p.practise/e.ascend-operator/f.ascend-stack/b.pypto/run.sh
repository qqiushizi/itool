#!/bin/bash
# ============================================================
# 实验: b.pypto (f.ascend-stack)
# 说明: PyPTO 张量编程:Python DSL + 自动算子生成
# 母目录: e.ascend-operator/f.ascend-stack
# 算子:   vector_add  (跨四栈对照基准算子)
# ============================================================
# 【第一性原理】
# PyPTO = 华为推出的 Python 算子开发框架, 用 Python 描述张量计算,
# 框架自动生成 AscendC / aclnn 后端代码。
# 核心概念:
#   1. Tensor        : N 维张量, 带 shape/dtype/layout
#   2. Tile          : 张量分块 (与 AscendC LocalTensor 对应, 装入 UB)
#   3. TileOp        : 对 Tile 的运算 (add/mul/matmul/exp...)
#   4. Kernel        : Python 函数, 描述一个算子的完整计算
#   5. InCore / OutCore : 核内 / 核外 tiling 策略
# 类比:
#   - PyPTO Kernel ~= AscendC __aicore__ kernel
#   - PyPTO Tile   ~= AscendC LocalTensor
#   - PyPTO Tensor ~= torch.Tensor (host 视角)
#   - PyPTO InCore ~= 显式 tiling + 双缓冲
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: b.pypto | PyPTO:Python DSL + 自动算子生成"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

# --- 1. PyPTO 心智模型 ---
hdr(1,TOTAL,"PyPTO 心智模型:Tensor / Tile / TileOp")
why("""PyPTO 把"写算子"拆成两层:
  - Kernel 层 : 描述算子语义 (输入/输出/Tile 大小)
  - TileOp 层 : 在 Tile 上做具体运算
开发者主要写 Kernel + TileOp, tiling/Pipe/同步由框架自动生成。""")
res("""三层结构:
  +----------------------------------------------+
  |  Python Kernel (用户写)                      |
  |    @pypto.kernel                             |
  |    def add(x, y):                            |
  |        tx = pypto.tile(x, [0, 0], [1024])    |  <-- Tile 切分
  |        ty = pypto.tile(y, [0, 0], [1024])    |
  |        tz = pypto.add(tx, ty)                |  <-- TileOp
  |        pypto.store(tz)                       |
  +----------------------------------------------+
                  |
                  v  框架自动生成
  +----------------------------------------------+
  |  AscendC Kernel (.cpp) + Tiling (.py)        |
  +----------------------------------------------+""")
mea("PyPTO 的价值: 用 Python 写算子, 自动生成 AscendC;\n  不用手动 Set/WaitFlag 也能得到 MTE/Compute overlap。")

# --- 2. 最小 vector_add Kernel ---
hdr(2,TOTAL,"最小 vector_add Kernel")
why("""vector_add 在 PyPTO 只需 5 行:""")
res('''import pypto
import pypto.language as pl

@pypto.kernel
def vector_add(x: pl.Tensor[pl.FP16, [N]],
               y: pl.Tensor[pl.FP16, [N]]) -> pl.Tensor[pl.FP16, [N]]:
    # 1) 在 N 维上切 1 块 (1 个 Tile, 装得下)
    tx = pl.tile(x, offsets=[0], shapes=[N])
    ty = pl.tile(y, offsets=[0], shapes=[N])
    # 2) TileOp: 逐元素加
    tz = pl.add(tx, ty)
    # 3) 输出
    return tz''')
mea("""3 个步骤:
  1. pl.tile(...)   : 把 Tensor 切出一个 Tile (落到 UB)
  2. pl.add(tx, ty) : TileOp, 框架转成 AscendC Add
  3. return tz      : 框架自动写回 HBM
无需手写: DataCopy、Set/WaitFlag、UB 分配 — 全部由编译器生成。""")

# --- 3. InCore 分块 ---
hdr(3,TOTAL,"InCore:大向量分多个 Tile 流水")
why("""向量长度 > UB 容量时, 必须切多 Tile。
PyPTO 的 pl.tile 支持多维偏移, 配 for 循环做 InCore 分块:""")
res('''@pypto.kernel
def vector_add_tiled(x: pl.Tensor[pl.FP16, [N]],
                     y: pl.Tensor[pl.FP16, [N]]) -> pl.Tensor[pl.FP16, [N]]:
    TILE = 4096
    out = pl.create_tensor([N], dtype=pl.FP16)
    for i in pl.range(0, N, TILE):           # 显式循环
        tx = pl.tile(x, offsets=[i], shapes=[TILE])
        ty = pl.tile(y, offsets=[i], shapes=[TILE])
        tz = pl.add(tx, ty)
        pl.store(out, tz, offsets=[i])
    return out''')
res_3 = """参数说明:
  - offsets=[i]   : Tile 在原 Tensor 的起始位置
  - shapes=[TILE] : Tile 形状 (N 不能整除时框架处理尾块)
  - pl.range(0,N,TILE) : 框架自动 unroll/pipeline""")
print("\n--- 补:参数说明 ---\n  " + res_3.replace("\n","\n  "))
mea("""InCore vs OutCore:
  - InCore  : 数据全在 UB 完成, 多次循环;适合 elementwise/reduction
  - OutCore : Tile 在 L2/HBM 间换入换出;适合 GEMM/Attention
  - pl.range 自动 unroll + double-buffer, 性能接近手写 AscendC""")

# --- 4. TileOp 一览 ---
hdr(4,TOTAL,"常用 TileOp 一览")
why("""PyPTO 提供与 AscendC 一一对应的 TileOp, 完整覆盖 Cube/Vector:""")
res("""分类      TileOp                              对应 AscendC
  元素    pl.add/sub/mul/div                    Add/Sub/Mul/Div
  激活    pl.relu/gelu/silu/exp/log/sqrt        Relu/Gelu/Exp/Log/Sqrt
  矩阵    pl.matmul(a, b)                       Matmul
  归约    pl.reduce_sum/max/min                 ReduceSum/Max/Min
  类型    pl.cast(x, dtype)                     Cast
  比较    pl.equal/less/where                   Eq/Lt/Where
  Shape   pl.reshape/transpose/concat           Reshape/Transpose/Concat
  搬运    pl.tile(src, off, shape)              DataCopy
  存储    pl.store(dst, tile, off)              DataCopy (回写)""")
mea("""TileOp 选型原则:
  - 能 pl.add 就不手写循环 (编译器优化更稳)
  - 矩阵乘必走 pl.matmul (走 Cube 单元)
  - 复杂融合 (LayerNorm) 用多个 TileOp 组合
  - 极端性能瓶颈才回退手写 AscendC""")

# --- 5. 编译 & 调用 ---
hdr(5,TOTAL,"编译/调用 & 与 AscendC 的关系")
why("""PyPTO 提供 3 种工作流:""")
res("""工作流:
  (A) JIT 调试:
      from pypto import runtime
      out = runtime.run(vector_add, x, y)

  (B) AOT 编译成 .so:
      pypto_aot vector_add.py -o libvec_add.so
      -> 生成 libvec_add.so (内部含 AscendC kernel)

  (C) 导出 AscendC 源码 (学习/二次开发):
      pypto_codegen vector_add.py --emit=cpp
      -> 生成 add_custom.cpp 等, 可读可改

  Python 调用:
      import pypto
      out = pypto.runtime.run(vector_add, x, y)
      # 或 torch_npu.op 注册后用 torch 调""")
mea("""PyPTO vs AscendC 取舍:
  +--------------------------------+----------------+
  | 维度            | PyPTO        | AscendC        |
  +--------------------------------+----------------+
  | 开发效率        | 高 (Python)  | 低 (C++)       |
  | 性能            | 接近手写     | 最优 (手调)    |
  | 调试            | 容易         | 难 (需 msprof) |
  | 灵活度          | 受限 TileOp  | 完全灵活       |
  | 学习曲线        | 平缓         | 陡             |
  | 适用场景        | 90% 算子     | 极致性能算子   |
  +--------------------------------+----------------+
  实战: 先用 PyPTO 写, 性能不够再下沉到 AscendC。""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:PyPTO = Python 写昇腾算子的 DSL;pl.tile 切块 + pl.add 等 TileOp 计算,
  框架自动生成 AscendC + tiling + 同步,不用碰 C++ 也能写出高效算子。
- 熟手:InCore 用 pl.range 切多 Tile,框架自动 unroll/double-buffer;
  TileOp 一一对应 AscendC API;JIT/AOT/codegen 三种工作流;性能不够时下沉 AscendC。
【进阶】写一个 pl.matmul 实现的 GEMM, 对比手写 AscendC 性能;用 pl.matmul+pl.bias+
pl.relu 实现 Fused Linear,验证融合收益。
EOF
echo "############################################################"
