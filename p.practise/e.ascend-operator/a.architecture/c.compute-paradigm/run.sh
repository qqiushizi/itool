#!/bin/bash
# ============================================================
# 实验: c.compute-paradigm
# 说明: 计算范式、同步、流水
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 算子开发三种范式:
#   1. 标量: 单条指令, 串行 (慢, 几乎不用)
#   2. SIMD: 一条指令处理多个数据 (Vector 单元)
#   3. SIMT: 单指令多线程 (CUDA 类, Ascend 也支持)
# 昇腾主要是 SIMD + 双发射, 强调:
#   - 数据并行 (SIMD-wide, 4096 bit)
#   - 任务并行 (Cube + Vector 同时跑)
#   - 数据搬运与计算流水 (MTE + Compute)
# 同步:
#   - Pipe 之间的 barrier (MTE2 → Compute → MTE3)
#   - SetFlag / WaitFlag 控制
# 流水 (Pipeline):
#   - 不同 iter 的同一阶段并行
#   - 双缓冲 / 多缓冲让 MTE 和 Compute 错开
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.compute-paradigm | SIMD + 双发射 + 流水 + 同步"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. SIMD vs SIMT vs 标量 ---
hdr(1,TOTAL,"SIMD vs SIMT vs 标量")
why("""三种计算范式:
  标量: 1 指令处理 1 数据, 简单但慢
  SIMD: 1 指令处理 N 数据 (向量化), Ascend Vector 单元 4096-bit
  SIMT: 1 指令多线程 (NVIDIA 风格), 灵活但调度复杂
  Ascend 偏向 SIMD + 显式双发射""")
res("""范式            抽象          硬件        编程模型
  标量           1条指令1数据   CPU 标量     顺序
  SIMD           1条指令N数据   Vector 4096-bit  AscendC::Vec
  SIMT           1条指令多线程  CUDA Core    threadIdx (CUDA)
  Ascend 整体:    SIMD 主 + SIMT 辅助 + 显式双发射""")
mea("""SIMD 优势:
  - 能效高 (1 个指令控制器, N 个 ALU 共用)
  - 易于融合
SIMT 优势:
  - 灵活 (warp 内可发散)
  - 适合复杂控制流
Ascend 选择 SIMD 是因为 AI 算子多是规整的 (matmul, conv), SIMD 效率最高。""")

# --- 2. 双发射 ---
hdr(2,TOTAL,"双发射:Cube 和 Vector 同时跑")
why("""双发射 = 一个 cycle 内, Cube 单元和 Vector 单元可同时各发一条指令。
例: Linear + Bias + ReLU:
  Cube:  算 C = A @ B      (用 Cube)
  Vector: 算 C = C + bias  (用 Vector)
  同时跑, 不串行
  加速比: 接近 1.5-2× (如果原本 Cube 时间 ≈ Vector 时间)""")
res("""无双发射 (串行):
  时间 = T(Cube: A@B) + T(Vec: +bias) + T(Vec: ReLU)
       = 10 + 2 + 2 = 14 us

有双发射 (理想):
  时间 = max(10, 2+2) = 10 us  (Cube 是瓶颈)
  加速: 1.4×""")
mea("""实战: 大多数 LLM 算子都是 Cube+Vector 双发。
  AscendC 写: 不用特别声明, 编译器自动尝试双发调度。
  但要: 数据准备好 (MTE 完成), Pipe 间有依赖。""")

# --- 3. 同步与流水 ---
hdr(3,TOTAL,"同步:5 个 Pipe 间的依赖")
why("""AI Core 有 5 个 Pipe:
  MTE1: HBM → L1
  MTE2: L1 → UB
  MTE3: UB → L1
  MTE4: L1 → HBM
  V/C:  Vector / Cube 计算
  
  同步 = 等某 Pipe 跑完再继续。
  AscendC API: SetFlag / WaitFlag / CrossCoreWaitFlag""")
res("""典型流水 (matmul):
  iter 0:  MTE2 (搬 A0,B0) → Cube (算 C0) → MTE3 (写 C0)
  iter 1:  MTE2 (搬 A1,B1) → Cube (算 C1) → MTE3 (写 C1)
  
  无双缓冲: 串行
  有双缓冲: iter 0 MTE2 完成后, iter 1 MTE2 立即开始
  
  同步 API:
  SetFlag<AscendC::HardEvent::MTE2_S> (event_id)
  WaitFlag<AscendC::HardEvent::MTE2_S> (event_id)""")
mea("同步写错 = 数据竞争 (race condition), 算子结果错但难调试。\n  AscendC 强制用 Set/WaitFlag, 不能简单加 fence。")

# --- 4. 流水性能 ---
hdr(4,TOTAL,"流水性能:从无流到双缓冲多缓冲")
why("""优化 3 个阶段, 性能逐步提升:""")
res("""阶段                 描述                         加速比
  无流水              搬 + 算 + 写 完全串行           1.0×
  单缓冲              上一 iter 算完再搬下一块         1.2-1.3×
  双缓冲              MTE 和 Compute 并行             1.5-1.8×
  多缓冲 (T3)         多 iter 流水线                  1.8-2.5×
  
  受限因素:
  - UB 大小: 多缓冲占 UB, 块大小受限
  - 算子类型: 计算和搬运哪个是瓶颈
  - 编译器: AscendC 编译器会尝试自动流水""")
mea("""实战经验:
  - 写简单算子: 单缓冲足够 (kernel 1.2-1.3× baseline)
  - 写大算子 (matmul/conv): 必须双缓冲以上
  - 极致性能: 多缓冲 (T3-pipeline) + 手写 schedule
  - 调试: 用 msprof 观察 pipe 利用率, 找出最闲的 pipe""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:计算范式 = 算子怎么用硬件;Ascend 走 SIMD + 双发射, 一个 cycle
  可以同时跑 Cube + Vector;同步用 SetFlag/WaitFlag;流水让搬运和计算并行。
- 熟手:多缓冲 (T3 pipeline) 是极致性能的标配;Cube+Vector 双发射可省 1.5-2×;
  Pipe 依赖错会 race condition, 难调试;msprof 看 pipe 利用率找瓶颈。
【进阶】用 AscendC 写一个 matmul 算子,对比无流水/双缓冲/多缓冲三版性能。
EOF
echo "############################################################"
