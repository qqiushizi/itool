#!/bin/bash
# ============================================================
# 实验: d.fusion-limits
# 说明: 融合限制:寄存器/UB 占用、计算密度、可融合性判断
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合不是\"越多越好\", 受 3 大限制:
#   1. 寄存器 / UB 占用: 多了装不下
#   2. 计算密度: 算子不能太重 (否则单 kernel 太慢)
#   3. 可融合性: 算子依赖关系决定能否融合
# 4 大不能融的情况:
#   1. 数据类型不一致 (FP16 + INT8) → cast
#   2. 需要同步 (跨 batch reduction) → 等全部完成
#   3. 数据流交叉 (gather/scatter) → 难融
#   4. UB 装不下整个数据 → 限制融合范围
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.fusion-limits | 限制:寄存器/UB/计算密度/可融合性"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. UB 限制 ---
hdr(1,TOTAL,"UB 限制:256 KB 是硬上限")
why("""融合后所有算子的数据都要装在 UB。
  例: Linear(1024→4096) + Bias + ReLU + Dropout
  A (1024, 1024) = 2 MB
  B (1024, 4096) = 8 MB
  out (1024, 4096) = 8 MB  ← 太大, 装不下
  → 必须 tile, 融合也要在 tile 内""")
out = ["  数据            大小       装得下 256KB?  处理"]
out.append("  tile A (128,128)  32 KB    是         OK")
out.append("  tile B (128,128)  32 KB    是         OK")
out.append("  tile out (128,128) 32 KB   是         OK")
out.append("  + 累加器          64 KB    是         OK")
out.append("  + 临时变量        32 KB    是         OK")
out.append("  合计              192 KB   是         OK")
out.append("  + 双缓冲          +192 KB  OOM!      错开 1 块")
res("\n".join(out))
mea("UB 256 KB 决定融合的\"小\"粒度:\n  - tile 大小上限\n  - 双缓冲最多 2 块\n  - 临时变量要省")

# --- 2. 计算密度 ---
hdr(2,TOTAL,"计算密度:kernel 不能太重")
why("""一个 kernel 内部:
  计算 T_calc = FLOPs / FLOPS_peak
  IO T_io = Bytes / BW_peak
  整体 T = max(T_calc, T_io) + T_sync
  
  如果融合后 T > 100 ms → 调度延迟, GPU 阻塞 → 不能融
  经验: 单 kernel < 1 ms 较合理""")
out = ["  算子            单次耗时  融合后      决策"]
out.append("  Linear 4096^3    50 us     50 us      OK (单算子)")
out.append("  + Bias          50+2 us    1 kernel  OK")
out.append("  + ReLU          50+4 us    1 kernel  OK")
out.append("  + Dropout       50+6 us    1 kernel  OK")
out.append("  + Cast          50+8 us    1 kernel  OK")
out.append("  + Softmax       50+30 us   1 kernel  风险 (kernel 80 us)")
out.append("  + LayerNorm     50+60 us   1 kernel  不建议 (kernel 110 us)")
res("\n".join(out))
mea("超重 kernel 风险:\n  1. 调度延迟, GPU 阻塞\n  2. 调试难, 出错难定位\n  3. 通用性差, 只能用于固定 shape\n建议: 融合后 kernel < 100 us")

# --- 3. 可融合性判断 ---
hdr(3,TOTAL,"可融合性:依赖关系图")
why("""判断两个算子能否融合:
  1. 算子 A 输出 = 算子 B 输入
  2. 没有中间同步 (如 all-reduce, 跨 batch)
  3. 数据类型一致 (或可隐式 cast)
  4. shape 一致 (或可 broadcast)""")
out = ["  算子对                   可融?   原因"]
out.append("  Linear -> ReLU           是      1-1 依赖, 同 dtype, 同 shape")
out.append("  Linear -> Softmax       部分    需 reduction 同步")
out.append("  Matmul -> AllReduce     否      跨卡同步")
out.append("  Cast (FP16->FP32) -> Add 是      1-1 依赖")
out.append("  Conv -> BN -> ReLU      是      经典融合, BN 几乎免费")
out.append("  LayerNorm -> Matmul     否      LN 需 reduction, 需同步")
res("\n".join(out))
mea("""融合判断流程:
  1. 画依赖图
  2. 检查同步点 (reduction, all-reduce, IO)
  3. 检查 dtype
  4. 检查 UB 容量
  5. 通过 → 融, 否则 → 不融""")

# --- 4. 实战取舍 ---
hdr(4,TOTAL,"实战:不该融的情况")
why("""6 种情况不要硬融:""")
out = ["  情况                原因                替代方案"]
out.append("  跨卡 all-reduce     通信开销大          不融, 单独算子")
out.append("  需 reduction       跨 block 同步       部分融, 后置归约")
out.append("  数据类型杂          cast 开销           统一后再融")
out.append("  kernel > 1 ms     太重, 调度延迟     拆成 2-3 个 fused kernel")
out.append("  shape 易变          通用性差            用框架动态 shape")
out.append("  调试中              难定位              先拆, 找到问题, 再融")
res("\n".join(out))
mea("""实战建议:
  1. 训练阶段不融 (shape 变化大)
  2. 推理阶段可大胆融 (shape 固定)
  3. 通信算子不融 (all-reduce)
  4. 跨 reduction 谨慎融
  5. 出 bug 时先拆, 定位后再融""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:融合有 3 大限制:UB 装不下, kernel 太重 (调度延迟), 算子不可融
  (跨 reduction, 跨卡);不融的情况: all-reduce、跨 reduction、shape 易变、
  dtype 不一致、kernel>1ms、调试中。
- 熟手:UB 256KB 决定融合粒度,单 kernel < 100us 较合理;依赖图判断可融合性;
  训练 shape 变不融,推理 shape 定可大胆融;通信算子永不融;msprof 看 fused
  kernel 时长避免超重。
【进阶】画一个 transformer block 的依赖图,标出哪些段可融,哪些段不能融;
  估算 fused 后总 IO 节省。
EOF
echo "############################################################"
