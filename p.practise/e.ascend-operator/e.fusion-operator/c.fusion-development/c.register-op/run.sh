#!/bin/bash
# ============================================================
# 实验: c.register-op
# 说明: 算子注册、图融合、buffer 分配
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 融合算子要上线, 必须完成 3 件事:
#   1. op_info.cfg: 告诉框架"我是什么算子"
#      - name, inputs/outputs, dtype, format
#      - 输入输出 shape 是否动态
#   2. 图融合 (graph fusion): 框架把多个 op 合成 1 个
#      - 自动算子融合 (auto fusion): 框架内置规则
#      - 手动图替换 (graph replace): pattern match
#   3. buffer 分配: 融合算子用更少 buffer
#      - 普通: 3 个算子 = 3 个中间 buffer
#      - 融合: 3 个算子 = 0 中间 buffer (in-place)
# 关键概念:
#   - 算子元信息 (op_info): 框架调度的依据
#   - 静态图 vs 动态图: 图融合在静态图才有效
#   - buffer 复用: 减少显存, 大模型关键
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: c.register-op | 算子注册、图融合、buffer 分配"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. op_info.cfg 配置 ---
hdr(1,TOTAL,"op_info.cfg:算子身份证")
why("""每个算子必须有 op_info.cfg, 框架靠它识别算子。""")
out = ["  字段            作用                       示例"]
out.append("  name           算子唯一名                  LinearReLUAdd")
out.append("  input_num      输入个数                   4 (X, W, B, Z)")
out.append("  output_num     输出个数                   1 (Y)")
out.append("  input_N        第 N 个输入 shape & dtype   (M,K) fp16")
out.append("  output_0       输出 shape & dtype         (M,N) fp16")
out.append("  dynamic_shape  是否支持动态 shape         true (可选)")
out.append("  build_options  编译选项                   -O2 --std=c++17")
res("\n".join(out))
mea("""op_info.cfg 写错 -> 算子调不通。
常见错误:
  1. shape 写死 -> 动态 shape 不支持
  2. dtype 不匹配 -> runtime error
  3. input 顺序反 -> 数据错位
  4. 漏 Build 依赖 -> 编译失败""")

# --- 2. 图融合 (Graph Fusion) ---
hdr(2,TOTAL,"图融合:框架自动优化")
why("""图融合 = 框架发现可融合的 op 序列, 自动换成 1 个融合 op。
两类:
  1. 算子融合 (Op Fusion): Linear+ReLU -> FusedLinearReLU
  2. 图融合 (Graph Fusion): 整段子图换成新算子
框架怎么知道能融合:
  - 数据流: A -> ReLU -> B, 中间无别的消费者
  - 类型匹配: ReLU 后只能是 ReLU 后允许的算子
  - 设备匹配: 都在 NPU 上""")
out = ["  融合类型     触发条件                  收益           风险"]
out.append("  Linear+ReLU   紧邻, 无分支             1.2-1.5x       精度对齐")
out.append("  Linear+LN    LN 紧跟, dim 匹配        1.5-2.0x       训练不稳")
out.append("  MHA(qkv)     3 个 matmul 紧邻         2.0-3.0x       长序列重")
out.append("  Softmax+MS   紧邻, 静态 shape         1.3-1.5x       mask 复杂")
out.append("  Concat+Split 互逆, 中间无算子         0 (免 buffer)  难写")
res("\n".join(out))
mea("""图融合的工程要点:
  1. 静态图才有效 (ONNX, TorchScript)
  2. 动态图 (PyTorch eager) 不融合
  3. 框架内置融合规则有限, 复杂融合要手写算子
  4. 融合后要 精度对齐 (用 fp32 算子对照)
  5. 融合后要 性能回归 (确保真的快了)""")

# --- 3. Buffer 分配优化 ---
hdr(3,TOTAL,"Buffer 分配:省显存的关键")
why("""Buffer 分配 = 算子中间结果的显存占用。
普通 3 个算子: Add -> ReLU -> Mul, 需要 2 个 buffer (ReLU 出一个, Mul 出一个)。
融合算子: FusedAddReLUMul, 0 个 buffer (in-place)。
大模型上 buffer 减少 -> 显存省 -> batch 大 / 序列长。""")
out = ["  算子序列                    中间 buffer 数   显存 (M=1024,d=1024)"]
out.append("  Add + ReLU + Mul (普通)     2 个            2 * 1024*1024*2 = 4 MB")
out.append("  Fused(Add+ReLU+Mul)        0 个            0 MB")
out.append("  7 层 Transformer 普通      ~20 个           ~40 MB")
out.append("  7 层 Transformer 全融合     ~5 个           ~10 MB")
out.append("  显存节省                    ~75%")
out.append("  推理 batch 收益:            batch 翻倍      30 GB -> 30 GB 装 2 倍 batch")
res("\n".join(out))
mea("""Buffer 优化的 3 个策略:
  1. 算子内 in-place: 累加器写回原 buffer
  2. 算子间 buffer 复用: 多个算子用同 1 个 buffer
  3. 算子融合: 0 中间 buffer
框架会做 (1) 和 (2), (3) 要手写融合算子""")

# --- 4. 注册完整流程 (伪代码) ---
hdr(4,TOTAL,"完整注册流程")
why("""从代码到框架调用, 完整流程:""")
res("""# 步骤 1: 写 op_info.cfg
# 文件: op_info.cfg
[LinearReLUAdd]
input_num = 4
output_num = 1
input_0 = (M,K), fp16
input_1 = (K,N), fp16
input_2 = (N,), fp16
input_3 = (M,N), fp16
output_0 = (M,N), fp16
build_options = -O2

# 步骤 2: 编译
cd project/
bash build.sh  # 调用 bisheng 编译,生成 .so
# 输出: build_out/liblinear_relu_add.so

# 步骤 3: 安装到 Ascend 包
cp build_out/liblinear_relu_add.so $ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe/custom/
cp op_info.cfg $ASCEND_HOME/opp/built-in/op_impl/ai_core/tbe/custom/config/

# 步骤 4: Python 端调用
import torch
# 算子已注册, 直接用
y = torch.ops.myops.linear_relu_add(x, w, b, z)

# 或者: 框架自动融合
y = torch.nn.functional.linear(x, w, b) + z  # 框架自动检测
y = torch.relu(y)""")
mea("""注册 4 步:
  1. op_info.cfg (必须)
  2. 编译 .so (bisheng)
  3. 安装到 Ascend 包
  4. Python 调用 (torch.ops 或框架自动)
调试技巧:
  - op_info.cfg 配错 -> 框架报 "not found"
  - .so 路径错 -> 框架报 "load failed"
  - 签名不匹配 -> runtime error
  - 第一次先在 CPU mock 验证, 再上 NPU""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:算子要上线必须 op_info.cfg (身份证) + 编译 + 安装 + Python 调用;
  图融合 = 框架自动把多个 op 合成 1 个, 静态图才有效;
  buffer 优化 = 算子融合后中间 buffer 减少, 显存省 30-75%。
- 熟手:op_info.cfg 写错就调不通, dtype/shape/format 必填;
  图融合触发条件: 紧邻 + 无分支 + dtype 匹配 + 设备同;
  buffer 3 策略: 算子内 in-place + 算子间复用 + 算子融合 (0 中间);
  复杂融合要手写算子, 内置规则覆盖不全。
【进阶】读 CANN 官方文档的 op_info.cfg 规范, 写一个自定义算子 (Linear+Softmax),
  完整走 编译 -> 安装 -> 静态图融合 -> 性能测试 流程, 对比融合前后的 buffer 占用。
EOF
echo "############################################################"
