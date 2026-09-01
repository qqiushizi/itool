#!/bin/bash
# ============================================================
# 实验: c.triton (f.ascend-stack)
# 说明: Triton:Python 写 GPU kernel,@triton.jit 自动编译
# 母目录: e.ascend-operator/f.ascend-stack
# 算子:   vector_add  (跨四栈对照基准算子)
# 备注:   Triton 原本面向 NVIDIA GPU;昇腾后端 (torch_npu + triton-ascend)
#         也在跟进,API 风格相同,本实验以通用 Triton 为主。
# ============================================================
# 【第一性原理】
# Triton = OpenAI 推出的 Python GPU 编程语言, 用 @triton.jit 装饰器
# 把 Python 函数编译成 GPU kernel (PTX/HIPCC/Ascend 后端)。
# 核心概念:
#   1. @triton.jit            : jit 装饰器, 函数体编译成 kernel
#   2. tl.tensor / tl.constexpr : 张量与编译期常量
#   3. program_id(0)          : 类似 CUDA blockIdx
#   4. tl.arange(0, BLOCK)    : 0..BLOCK-1 的索引向量 (在 SM/UB 上)
#   5. BLOCK_SIZE             : 编译期 tile 大小 (装入 SMEM/UB)
# 类比:
#   - @triton.jit fn   ~= __global__ kernel
#   - tl.tensor        ~= LocalTensor (SMEM/UB)
#   - program_id       ~= blockIdx
#   - BLOCK_SIZE       ~= tile 长度
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: c.triton | Triton:@triton.jit / BLOCK / autotune"
echo "############################################################"

python3 <<"PYEOF"
import numpy as np
def hdr(n,total,t): print(f"\n{"="*60}\n【步骤 {n}/{total}】{t}\n{"="*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

# --- 1. Triton 心智模型 ---
hdr(1,TOTAL,"Triton 心智模型:program / block / tile")
why("""Triton 把 GPU 编程简化为 3 个概念:
  - program (类似 CUDA block): 一个 program 处理一块数据
  - BLOCK_SIZE: 每个 program 处理的元素数 (装入 SMEM/UB)
  - num_warps / num_stages   : 编译期微架构参数""")
res("""对应关系:
  Triton                    CUDA                 昇腾 AscendC
  @triton.jit fn            __global__ fn        __aicore__ fn
  program_id(axis)          blockIdx             GetBlockIdx
  tl.arange(0, BLOCK)       threadIdx.x          LocalTensor index
  tl.load(ptr)              ld.global            DataCopy (HBM->UB)
  tl.store(ptr, v)          st.global            DataCopy (UB->HBM)
  BLOCK_SIZE                blockDim.x * vec     TILE_LEN
  num_warps                 --                   (对应 Vector 宽度)
  num_stages                --                   双/多缓冲""")
mea("Triton 在"写"和"调"两端都极简: 写 = Python 数组风格, 调 = autotune 自动选参。")

# --- 2. 最小 vector_add kernel ---
hdr(2,TOTAL,"最小 vector_add: 5 行 kernel")
why("""Triton 不区分 HBM/UB, 概念上是"指针 + tile 计算":""")
res("""import triton
import triton.language as tl

@triton.jit
def vec_add_kernel(x_ptr, y_ptr, z_ptr, N,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                # 当前 program id
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)    # 本 block 负责的索引
    mask = offs < N                                       # 边界 mask
    x = tl.load(x_ptr + offs, mask=mask)                  # 从 HBM 加载 tile
    y = tl.load(y_ptr + offs, mask=mask)
    z = x + y                                             # tile 计算
    tl.store(z_ptr + offs, z, mask=mask)                  # 写回 HBM

def vec_add(x, y):
    z = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)                       # grid = ceil(N/BLOCK)
    vec_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK)
    return z""")
mea("""5 个必备元素:
  - @triton.jit            : 编译成 GPU kernel
  - tl.program_id(0)       : 当前 program 编号
  - tl.arange(0, BLOCK)    : 本 program 内的 tile 索引
  - tl.load / tl.store     : 显式 HBM 读写
  - mask=                  : 边界保护 (处理 N 不能整除 BLOCK)""")

# --- 3. Autotune: 自动选最佳 BLOCK_SIZE ---
hdr(3,TOTAL,"Autotune:自动搜索最佳配置")
why("""BLOCK_SIZE / num_warps / num_stages 难以手工选, 用 @triton.autotune:""")
res("""@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def vec_add_auto(x_ptr, y_ptr, z_ptr, N,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)

# 第一次调用会 benchmark 所有 config, 之后缓存最优配置
vec_add_auto[lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)](x, y, z, N)""")
mea("""Autotune 用法:
  - configs      : 候选 (BLOCK_SIZE, num_warps, num_stages, ...) 组合
  - key          : 哪些输入参数决定"同组配置" (如 N, dtype)
  - cache        : 第一次选好后存盘, 后续直接用
  - 实战        : GEMM/Attention 用 autotune 收益最大 (3-5x)""")

# --- 4. 常用 Tile 级 API ---
hdr(4,TOTAL,"常用 Tile 级 API")
why("""Triton 提供与 NumPy 类似的 tile 级运算:""")
res("""类别      API                                      说明
  元素    tl.add / sub / mul / div / neg             逐元素
  激活    tl.exp / log / sqrt / rsqrt / relu        数学/激活
  矩阵    tl.dot(a, b)                              矩阵乘 (TensorCore / Cube)
  归约    tl.sum / max / min / mean (axis=)         沿轴归约
  类型    .to(tl.float16)                           类型转换
  比较    tl.where(cond, a, b)                      条件选择
  Shape   tl.reshape / tl.trans / tl.broadcast_to  Shape
  索引    tl.arange / tl.load(..., mask=)           显式索引
  原子    tl.atomic_add                             原子操作""")
mea("""对比 AscendC:
  - tl.dot ~= pl.matmul ~= AscendC::Matmul
  - tl.sum  ~= pl.reduce_sum ~= AscendC::ReduceSum
  - 复杂融合: layer_norm/softmax Triton 都有官方 example
  - 矩阵乘 + 激活的融合是 Triton 强项 (写起来 20 行)""")

# --- 5. 编译 & 跨后端 ---
hdr(5,TOTAL,"编译 & 跨后端 (CUDA / ROCm / Ascend)")
why("""Triton 一次写, 多后端编译;昇腾上使用方式:""")
res("""后端       安装                          后端路径
  NVIDIA   pip install triton              默认 CUDA / PTX
  AMD      pip install triton-rocm         HIP / CDNA
  昇腾     pip install triton-ascend       AscendC 目标 (CANN)

  实战 (NVIDIA + 昇腾 混合):
    import triton
    @triton.jit
    def my_kernel(...): ...
    # 代码 0 改动, 在不同 device 上跑

  Ascend 后端说明:
    - 由 torch_npu 团队维护, 部分高级特性 (如 dot) 成熟
    - 元素级 / 简单 matmul 已可生产
    - 极致性能算子仍建议手写 AscendC""")
mea("""Triton 取舍:
  +----------------------------+----------------------+
  | 优点                       | 缺点                 |
  +----------------------------+----------------------+
  | Python 写, 不用 C++        | 极致性能不如手写     |
  | autotune 自动选参          | 控制粒度较粗         |
  | 跨后端 (CUDA/ROCm/Ascend)  | 调试信息弱           |
  | 与 torch 集成好             | 后端成熟度差异大     |
  +----------------------------+----------------------+
  实战: 新算子原型阶段用 Triton 写, 验证 idea, 性能是瓶颈再下沉 AscendC。""")
PYEOF

echo ""
echo "############################################################"
cat <<"EOF"
【整体总结】
- 小白:Triton = Python 写 GPU kernel;@triton.jit 装饰函数, tl.load/store 显式访存,
  BLOCK_SIZE 控制 tile 大小,autotune 自动选最佳配置;NVIDIA/AMD/昇腾多后端。
- 熟手:tl.dot 走 TensorCore/Cube, tl.sum/max 归约;num_warps/num_stages 微架构调优;
  @triton.autotune 在 GEMM/Attention 上能提速 3-5x;Ascend 后端成熟度低于 NVIDIA。
【进阶】用 @triton.autotune 写一个 GEMM, 对比 cublas / AscendC::Matmul;尝试把
PyTorch 模型的某个 elementwise+reduction 算子用 Triton 重写, 看端到端加速。
EOF
echo "############################################################"
