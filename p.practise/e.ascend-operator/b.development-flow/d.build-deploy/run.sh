#!/bin/bash
# ============================================================
# 实验: d.build-deploy
# 说明: 编译、部署、aclnn 接口
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 昇腾算子从源码到上线 5 步:
#   1. 写 Ascend C 代码 (.cpp)
#   2. 编译 (bisheng -> 字节码)
#   3. 链接 (clang + CANN runtime -> .so)
#   4. 注册 (通过 \"自定义算子\" 机制注册到 CANN)
#   5. 调用 (aclnnXxx 或 torch_npu)
# 部署:
#   - 单卡: 直接 aclnn
#   - 多卡: aclnn + HCCL 通信
#   - 模型集成: 注册为 torch 算子, 在 PyTorch 中调用
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: d.build-deploy | 编译 + 部署 + aclnn 调用"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,total,t): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 编译流程 ---
hdr(1,TOTAL,"编译:从 .cpp 到 .so")
why("""Ascend C 算子编译分两步:
  1. bisheng: .cpp -> .o (昇腾后端编译, 生成 AICore 字节码)
  2. clang++: .o + runtime -> .so (链接)
  注意: bisheng 编译慢 (5-10 分钟), 增量编译只重编改过的 .cpp""")
res("""完整编译命令:
  # 1. 编译为 .o
  $CANN_HOME/bin/bisheng \\
    --cce-aicore-arch=davinci-200 \\
    -O2 -std=c++17 \\
    -I $CANN_HOME/include \\
    -I $CANN_HOME/include/acl \\
    -I $CANN_HOME/include/aclnn \\
    my_op.cpp -o my_op.o

  # 2. 链接为 .so
  $CANN_HOME/bin/clang++ \\
    -shared -fPIC \\
    -L $CANN_HOME/lib64 \\
    my_op.o -lascendcl -lruntime -o libmy_op.so

  # 3. 安装到 ops 目录
  cp libmy_op.so $ASCEND_HOME/ops/my_op/""")
mea("""常见编译问题:
  - 'ascendcl.h' not found: 检查 CANN_HOME 环境变量
  - 链接错: 缺 -lascendcl
  - 字节码 arch 不匹配: --cce-aicore-arch 选对 (davinci-100/200/...)
  - 编译慢: 用 ccache 缓存""")

# --- 2. 算子注册 ---
hdr(2,TOTAL,"算子注册:让 aclnn 能找到")
why("""自定义算子需要\"注册\"才能被 aclnn 找到。
  注册方式:
    1. JSON 配置文件: op_info.cfg
    2. python setup 脚本
    3. 集成到 torch_npu (PyTorch)
  注册 = 告诉 CANN: 这个算子叫啥、入参是啥、输出是啥""")
res("""op_info.cfg 示例:
  {
    \"op\": \"MyAddCustom\",
    \"input_desc\": [
      {\"name\": \"x\", \"type\": \"float\", \"format\": \"ND\"},
      {\"name\": \"y\", \"type\": \"float\", \"format\": \"ND\"}
    ],
    \"output_desc\": [
      {\"name\": \"z\", \"type\": \"float\", \"format\": \"ND\"}
    ]
  }

  # 编译
  $CANN_HOME/bin/asc_op_compiler \\
    --input=my_op.cpp \\
    --output=./output \\
    --op_info=op_info.cfg

  # 安装
  ./output/run_my_op.sh  # 自动拷贝到 op 目录""")
mea("""注册后调用:
  import torch
  z = torch.ops.myops.MyAddCustom(x, y)  # torch 视角
  或
  aclnnMyAddCustom(x, y, z, ...)  # aclnn 视角""")

# --- 3. 部署形态 ---
hdr(3,TOTAL,"3 种部署形态")
why("""昇腾算子部署 3 种形态:""")
res("""形态           适用                       流程
  1. aclnn 库    torch_npu / MindSpore 自动   编译 .so + 注册 → 直接用
  2. CANN APP    独立 C++ app, 调 aclnn        链接 CANN runtime
  3. 自研引擎    类似 vLLM / DeepSeek         集成 torch_npu op
""")
mea("""实际路径:
  - 90% 用户: torch_npu 已封装所有主流算子, 无需自研
  - 性能敏感: 手写 Ascend C, 注册为 torch 算子
  - 国产化项目: 必经流程, 编译部署是日常工作""")

# --- 4. 部署检查清单 ---
hdr(4,TOTAL,"部署前 7 项检查")
why("""算子部署到生产前必查:""")
res("""检查项                    工具
  1. 编译通过                bisheng + clang
  2. 链接通过                clang++ -shared
  3. 注册成功                op_info.cfg
  4. 数值正确 (vs torch)     随机数据对比
  5. 边界正确 (边界 tile)    极端 size 测试
  6. 性能达标 (vs baseline)  msprof
  7. 内存正确 (无 leak)     valgrind / msprof""")
mea("""上线流程:
  1. 单卡验证: 编译 + 数值正确
  2. 多卡验证: HCCL 通信正确
  3. 性能基线: vs 标杆算子 (e.g. CANN 内置)
  4. 灰度发布: 5% 流量, 看业务指标
  5. 全量: 无问题再 100%
  
  关键: 不要等上线才测, 写算子时就考虑边界 + 性能""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:算子部署 = 编译 (.cpp -> .so) + 注册 (op_info.cfg) + 调用
  (aclnn 或 torch);bisheng + clang 编译;生产前必做数值 + 性能 + 边界检查。
- 熟手:编译慢用 ccache;自定义算子注册 JSON 配置;多卡部署加 HCCL;
  msprof 调优;torch_npu 已封 90% 算子, 极致性能才自研;灰度上线。
【进阶】自己写一个简单 add 算子, 完整走 编译 -> 注册 -> Python 调用流程。
EOF
echo "############################################################"
