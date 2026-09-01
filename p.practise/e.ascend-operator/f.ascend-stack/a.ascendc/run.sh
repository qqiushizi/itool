#!/bin/bash
# ============================================================
# 实验: a.ascendc (f.ascend-stack)
# 说明: AscendC 算子开发:kernel/tiling/双缓冲/Pipe 同步
# 母目录: e.ascend-operator/f.ascend-stack
# 算子:   vector_add  (跨四栈对照基准算子)
# ============================================================
# 【第一性原理】
# AscendC 是华为官方为昇腾 NPU 提供的 C++ 算子开发语言,
# 直接面向 AICore 的 5 段流水线 (MTE1/MTE2/Cube/Vector/MTE3)。
# 编程范式:
#   1. __global__ __aicore__ void kernel(...)  入口
#   2. __gm__ uint8_t*  GlobalTensor (HBM 端)
#   3. __local__  / AscendC::LocalTensor  (UB 端,256KB)
#   4. DataCopy (MTE) + Add/Sub/Mul/... (Vector/Cube)
#   5. SetFlag / WaitFlag<PIPE_XXX> 控制 5 段流水线依赖
# 性能优化三大手段:
#   - 合理 tile 大小 (装得进 UB, 越大复用越好)
#   - 双缓冲 / 多缓冲 (MTE 与计算 overlap)
#   - Vector/Cube 双发射 (同指令周期同时跑)
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: a.ascendc | AscendC:kernel/tiling/双缓冲/同步"
echo "############################################################"

python3 <<'PY_INNER'
import numpy as np
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 5

hdr(1,TOTAL,"AscendC 算子文件结构 (vector_add)")
why("""一个最小 AscendC 算子由 3 部分组成:
  - add_custom.cpp  : AICore kernel 实现
  - add_custom.py   : host 端封装 (Tiling + launch)
  - CMakeLists.txt  : 用 AscendC 提供的 cmake 宏编译""")
res("""目录布局:
  op_add/
    +- add_custom.cpp        # AICore kernel
    +- add_custom.py         # host 端 (tiling + 启动)
    +- CMakeLists.txt        # 编译配置
    +- scripts/
        +- gen_tiling.py     # tiling 模板生成""")
mea("AscendC 工程化 = kernel (.cpp) + tiling (.py) + 编译脚本;\n  跟 CUDA 的 .cu + .cpp + setup.py 是同一思路。")

hdr(2,TOTAL,"AICore kernel: vector_add 完整实现")
why("下面是一个生产可用的 AscendC vector_add kernel, 展示 4 个必备元素 + tiling 拆分 + UB 复用:")
res("""#include "kernel_operator.h"
using namespace AscendC;

constexpr int BUFFER_NUM = 1;       // 单缓冲 (demo 用)
constexpr int TILE_LEN   = 4096;    // 单 tile 元素数

extern "C" __global__ __aicore__ void add_custom(
    __gm__ float* x, __gm__ float* y, __gm__ float* z,
    uint32_t total_elem) {

  uint32_t block_idx = GetBlockIdx();
  uint32_t block_num = GetBlockNum();
  uint32_t per_block = (total_elem + block_num - 1) / block_num;
  uint32_t start = block_idx * per_block;
  uint32_t end   = min(start + per_block, total_elem);
  uint32_t len   = end - start;

  LocalTensor<float> lx(BUFFER_NUM, TILE_LEN);
  LocalTensor<float> ly(BUFFER_NUM, TILE_LEN);
  LocalTensor<float> lz(BUFFER_NUM, TILE_LEN);

  uint32_t i = 0;
  for (; i + TILE_LEN <= len; i += TILE_LEN) {
    DataCopy(lx, x + start + i, TILE_LEN);
    DataCopy(ly, y + start + i, TILE_LEN);
    SetFlag<PIPE_MTE2>(0); WaitFlag<PIPE_MTE2>(0);
    Add(lz, lx, ly, TILE_LEN);
    SetFlag<PIPE_V>(0);    WaitFlag<PIPE_V>(0);
    DataCopy(z + start + i, lz, TILE_LEN);
    SetFlag<PIPE_MTE3>(0); WaitFlag<PIPE_MTE3>(0);
  }
  if (i < len) {
    uint32_t tail = len - i;
    DataCopy(lx, x + start + i, tail);
    DataCopy(ly, y + start + i, tail);
    SetFlag<PIPE_MTE2>(0); WaitFlag<PIPE_MTE2>(0);
    Add(lz, lx, ly, tail);
    SetFlag<PIPE_V>(0);    WaitFlag<PIPE_V>(0);
    DataCopy(z + start + i, lz, tail);
    SetFlag<PIPE_MTE3>(0); WaitFlag<PIPE_MTE3>(0);
  }
}""")
mea("""4 个必备元素:
  - __global__ __aicore__     : kernel 入口标记
  - __gm__ float*             : GlobalTensor (HBM)
  - LocalTensor<float>        : UB 上的局部张量
  - DataCopy + Add            : MTE 搬运 + Vector 计算
尾块循环: 处理 total 不能被 tile 整除的边界。""")

hdr(3,TOTAL,"Tiling 策略:大向量切块")
why("AscendC 没有 grid 维度概念, 由 host 端通过 tiling 参数告诉 kernel: total_elem / tile_len / block_num")
res("""Tiling 计算示例 (total=1M 元素, A2 64 核):
  tile_len   = 4096           # 16 KB FP32, 占 UB 极少
  block_num  = 64             # A2 单 device 64 核
  per_block  = ceil(1M / 64)  = 16384
  循环次数/核 = 16384 / 4096  = 4 次
  UB 占用    = 3 * 4096 * 4B  = 48 KB  (远小于 256KB)""")
mea("""Tiling 经验:
  - tile_len 不超 UB/2 (留双缓冲空间)
  - block_num 取 device 实际核数 (A2=64, A3=?)
  - 大向量尽量按一核一段 拆, 减少边界尾块
  - 用 msprof/tikernel 工具看 MTE 占比确认是否合理""")

hdr(4,TOTAL,"双缓冲:让 MTE 与 Compute overlap")
why("""单缓冲:MTE 等 Compute, Compute 等 MTE (串行)
双缓冲:MTE 搬 buffer[ping] 时, Compute 算 buffer[pong]
       -> MTE/Compute 流水, 理论 1.5-2x 加速""")
res("""伪代码:
  for (i = 0; i < N; i += 2*TILE) {
    DataCopy(lx_ping, gm_x + i, TILE);
    DataCopy(ly_ping, gm_y + i, TILE);
    SetFlag<PIPE_MTE2>(0); WaitFlag<PIPE_MTE2>(0);
    if (i + TILE < N) {
      DataCopy(lx_pong, gm_x + i + TILE, TILE);
      DataCopy(ly_pong, gm_y + i + TILE, TILE);
      SetFlag<PIPE_MTE2>(1); WaitFlag<PIPE_MTE2>(1);
    }
    Add(lz_ping, lx_ping, ly_ping, TILE);
    SetFlag<PIPE_V>(0); WaitFlag<PIPE_V>(0);
    DataCopy(gm_z + i, lz_ping, TILE);
    SetFlag<PIPE_MTE3>(0); WaitFlag<PIPE_MTE3>(0);
  }
  占用 UB: 3 * 2 * TILE * 4B ~ 96 KB (A2 256KB 内 OK)""")
mea("""双缓冲是 AscendC 性能优化的关键:
  - MTE 带宽高时, MTE 隐在 Compute 后面, 端到端 ~= max(MTE, Compute)
  - 三缓冲 / 多缓冲: GEMM 等计算密集场景用 (MTE 多次 overlap)
  - elementwise 计算轻, 双缓冲一般足够
  - 用 msprof 对比单/双缓冲耗时验证""")

hdr(5,TOTAL,"编译 & 集成到 torch_npu")
why("AscendC 算子最终要注册到 torch_npu / aclnn 才能被 Python 调用:")
res("""# 1) 编译 .cpp -> .o
$CANN_HOME/bin/bisheng \\
  --cce-aicore-arch=davinci-220 \\
  -O2 -std=c++17 \\
  -I$CANN_HOME/include \\
  add_custom.cpp -o add_custom.o

# 2) 链接 .o + runtime -> .so
$CANN_HOME/bin/clang++ \\
  add_custom.o -lascendcl -shared -fPIC \\
  -o libadd_custom.so

# 3) 通过 torch_npu 自定义算子机制注册 (TORCH_LIBRARY / aclnn)
# 4) Python 调用
import torch, torch_npu
x = torch.randn(1_000_000, device='npu')
y = torch.randn(1_000_000, device='npu')
z = torch.ops.myops.add_custom(x, y)""")
mea("""常见问题:
  - 编译慢: 用 ccache / 缓存 .o
  - 链接错: 缺 $CANN_HOME 环境变量
  - OOM: tile_len 太大 -> 调小或加多缓冲
  - 数据错: 漏 Set/WaitFlag -> 加同步
  - 性能差: msprof 看 MTE/Vector 占比 -> 加双缓冲""")
PY_INNER

echo ""
echo "############################################################"
cat <<'SUMMARY_END'
【整体总结】
- 小白:AscendC = 昇腾 NPU 的 C++ 算子语言;__aicore__ 写 kernel,__gm__/LocalTensor
  区分 HBM/UB;DataCopy 搬运 + Add 等 API 计算;用 bisheng+clang 编译成 .so 注册。
- 熟手:5 段流水线显式同步 (Set/WaitFlag);Tiling 拆 4096 元素/tile,UB 占用 < 256KB;
  双缓冲让 MTE/Compute overlap 提速 1.5-2x;Cube/Vector 双发射可同时跑。
【进阶】读 CANN samples/operator/AddCustom;用 msprof 对比单/双缓冲;把 add 升级
为 add+LeakyReLU 融合算子,看 GEMM+Activation 端到端提升。
SUMMARY_END
echo "############################################################"
