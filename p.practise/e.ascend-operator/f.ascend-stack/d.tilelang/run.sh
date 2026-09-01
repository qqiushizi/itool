#!/bin/bash
# ============================================================
# 实验: d.tilelang (f.ascend-stack)
# 说明: TileLang:Tile 优先的 Python DSL, 多后端 (CUDA/HIP/AscendC)
# 母目录: e.ascend-operator/f.ascend-stack
# 算子:   vector_add  (跨四栈对照基准算子)
# ============================================================
# 【第一性原理】
# TileLang = 上海 AI Lab 推出的 Tile 优先 Python DSL, 把 tile 切分 + buffer 复用 +
# 后端代码生成做成显式原语, 旨在比 Triton 更接近硬件, 而比手写 CUDA/AscendC 更高产。
# 核心概念:
#   1. T.K / T.V / T.macro       : 维度/变量/宏的声明
#   2. T.tile(dom)               : 显式 tile 视图 (与 AscendC LocalTensor 对应)
#   3. T.alloc_buffer(...)       : 显式分配 buffer (可控在 HBM/UB/SMEM)
#   4. T.copy(...) / T.compute   : 显式搬运与计算
#   5. T.Kernel(...) / T.grid    : 启动配置
# 设计哲学:
#   - Tile 是一等公民, 跟硬件 tile 大小显式绑定
#   - 让用户能精细控制 memory/layout, 又比 AscendC 简洁
#   - 跨后端: 一份 TileLang 生成 CUDA / AscendC / HIP
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.tilelang | TileLang:T.tile / T.alloc_buffer / T.copy"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

hdr(1,TOTAL,"TileLang 心智模型:Tile / Buffer / Compute 分离")
why("""TileLang 跟 Triton 最大区别:
  - Triton: tile 计算 + load/store 隐式 (编译器决定访存)
  - TileLang: tile / buffer / copy / compute 显式 (用户精细控制)
  -> 性能上限更高, 学习曲线略陡, 但仍是 Python""")
res("""统一心智模型:
  +-----------------+   4 类原语   +--------------------+
  | Python Kernel   | ----------> | T.tile / T.alloc    |
  |                 |             | T.copy  / T.compute |
  +-----------------+             +--------------------+
                                       |
                                       v  编译器
                  +-------------------+--------------------+
                  |                                        |
            CUDA / HIPCC                          AscendC (CANN)
            (NVIDIA / AMD)                          (昇腾 NPU)""")
mea("TileLang 的目标: 在 Python 层拿到『接近手写 AscendC』的控制力。")

hdr(2,TOTAL,"最小 vector_add Kernel")
why("TileLang 显式 4 阶段: alloc -> copy -> compute -> copy out")
res("""import tilelang.language as T

def vector_add(M, N, dtype="float16", block_M=1024):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),    # HBM
        B: T.Tensor((M, N), dtype),    # HBM
        C: T.Tensor((M, N), dtype),    # HBM
    ):
        with T.Kernel(T.ceildiv(N, block_M), is_npu=True) as bx:
            a = T.alloc_buffer((block_M,), dtype, scope="shared")
            b = T.alloc_buffer((block_M,), dtype, scope="shared")
            c = T.alloc_buffer((block_M,), dtype, scope="shared")
            T.copy(A[0, bx * block_M], a)
            T.copy(B[0, bx * block_M], b)
            T.compute(c, lambda i: a[i] + b[i])
            T.copy(c, C[0, bx * block_M])
    return main""")
mea("""5 个必备阶段:
  - T.Kernel(grid, is_npu=...)   : 启动 (bx = program id)
  - T.alloc_buffer(..., scope=) : 显式 UB/SMEM buffer
  - T.copy(src, dst)             : 显式搬运
  - T.compute(dst, lambda)       : 显式计算 (lambda 风格)
  - T.copy 回写                  : 输出
scope='shared' 在 NVIDIA 走 SMEM, 昇腾后端会映射到 UB。""")

hdr(3,TOTAL,"Buffer 复用 & 多级内存 (shared/UB vs global/HBM)")
why("""TileLang 的 scope 参数是性能关键:
  - 'global'   : HBM/DRAM, 慢, 大
  - 'shared'   : SMEM/UB,  快, 小
  - 'register' : 寄存器, 最快, 极小
手动管理 buffer 复用, 是 TileLang 比 Triton 更细的地方。""")
res("""复用示例 (GEMM 片段):
  with T.Kernel(M//bm, N//bn, is_npu=True) as (mx, nx):
    A_s = T.alloc_buffer((bm, bk), dtype, scope='shared')  # A tile
    B_s = T.alloc_buffer((bk, bn), dtype, scope='shared')  # B tile
    acc = T.alloc_buffer((bm, bn), 'float32', scope='local')  # 累加器
    for ko in T.serial(K // bk):
        T.copy(A[mx*bm, ko*bk], A_s)
        T.copy(B[ko*bk, nx*bn], B_s)
        T.compute(acc, lambda i, j: acc[i,j] + A_s[i,ko] * B_s[ko,j])
    T.copy(acc, C[mx*bm, nx*bn], cast=dtype)

  内存层级:  global (HBM)  ->  shared (UB/SMEM)  ->  register
  数据复用:  A/B 一次加载, 多次 MAC, 提升 arithmetic intensity""")
mea("""Buffer 设计原则:
  - 频繁复用的小块放 shared/UB
  - 累加器放 register/local (fp32)
  - 跨循环不变的只读块放 shared, 不复用放 global
  - TileLang 的 scope= 是性能调优的核心开关""")

hdr(4,TOTAL,"常用原语速查")
why("TileLang 原语按『生命周期』分组:")
res("""阶段         原语                                用途
  声明       T.Kernel(grid, is_npu)               启动 grid
             T.alloc_buffer(shape, dtype, scope)  显式 buffer
  搬运       T.copy(src, dst)                     HBM<->shared/UB
             T.copy(..., coalesced=True)          合并访存
  计算       T.compute(dst, lambda)               lambda 计算
             T.gemm(...) / T.matmul(...)          矩阵乘 (生成 Cube/Matmul)
             T.reduce(dst, src, axis=)            归约
  Shape      T.reshape / T.trans / T.expand       形状变换
  控制       T.serial(N) / T.parallel(N)          循环并行性
  同步       T.fence(scope) / T.barrier()         buffer fence / 跨核""")
mea("""对比 AscendC:
  - T.alloc_buffer(..., scope='shared') ~= LocalTensor
  - T.copy                     ~= DataCopy
  - T.compute(lambda)          ~= 元素级 Vector API
  - T.gemm                     ~= AscendC::Matmul (Cube)
  - T.fence                    ~= Set/WaitFlag
TileLang 写 GEMM 比 AscendC 短 5-10x, 性能接近手写。""")

hdr(5,TOTAL,"后端 & 编译 (CUDA / HIP / Ascend)")
why("TileLang 多后端, 昇腾后端由 CANN 团队提供:")
res("""安装:
  pip install tilelang               # CPU/NVIDIA 默认
  pip install tilelang-ascend        # 昇腾 NPU 后端

调用:
  # 1) AOT 编译成 kernel
  kernel = tilelang.compile(vector_add, target='ascend')  # or 'cuda'
  # 2) Python 调用 (torch_npu 集成)
  out = kernel(x, y)

Ascend 后端成熟度:
  - 元素级 / 简单 GEMM: 可生产
  - 复杂 fusion: 部分支持, 看 release notes
  - 性能 hand-tuning 空间: 大""")
mea("""TileLang 定位:
  - 介于『Python 易用』和『AscendC 极致性能』之间
  - 想细控 buffer/访存又想用 Python 写时优先选
  - 适合: GEMM/Attention/LayerNorm 这类访存密集的 fusion
  - 不适合: 极轻量 elementwise (Triton 更省事)""")
PYEOF

echo ""
echo "############################################################"
cat <<'TILELANG_EOF'
【整体总结】
- 小白:TileLang = Tile 优先的 Python DSL;T.Kernel 启动, T.alloc_buffer 显式 buffer,
  T.copy 搬运 + T.compute 计算;scope=shared/UB 映射到硬件;跨 NVIDIA/昇腾后端。
- 熟手:scope= 显式控制 memory hierarchy;T.gemm/T.fence 对应 Cube/MTE 同步;
  比 Triton 更细的 buffer 复用控制, 比 AscendC 写起来短 5-10x;昇腾后端在快速成熟。
【进阶】用 TileLang 写一个 fused LayerNorm (归约+减+除+gamma/beta),
  对比手写 AscendC 和 torch.layer_norm 性能;尝试把 Transformer Attention 用
  TileLang 重写, 看 FlashAttention 风格实现。
TILELANG_EOF
echo "############################################################"
