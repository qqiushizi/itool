#!/bin/bash
# ============================================================
# 实验: a.hang-debug
# 说明: 集合通信 hang (卡死) 调试与解决
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# 集合通信 hang = 卡死, 不退出, 不报错, 也不继续。
# 三大原因:
#   1. 死锁: A 等 B, B 等 A, 永循环
#   2. 失配: 不同 rank 调不同的原语
#   3. 网络问题: 节点不通, IB 断, RDMA 失败
# 排查思路:
#   1. 看 NCCL_DEBUG=INFO 日志
#   2. 看 rank 表, IP/端口对不对
#   3. 看防火墙, 端口开没开
#   4. 用 NCCL_DEBUG_SUBSYS=ALL 详细日志
#   5. 用 strace/gdb 抓 stack
#   6. 用 torch.distributed.monitor 监控
# 工具:
#   - NCCL_DEBUG=INFO/WARN/TRACE
#   - NCCL_DEBUG_SUBSYS=ALL
#   - NCCL_ASYNC_ERROR_HANDLING
#   - NCCL_TIMEOUT (默认 30 min)
# ============================================================
set -euo pipefail
echo "############################################################"
echo "# 实验: a.hang-debug | 集合通信 hang 调试"
echo "############################################################"

python3 <<'PYEOF'
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. Hang 的 3 大原因 ---
hdr(1,TOTAL,"Hang 的 3 大原因")
why("""集合通信卡死, 3 大原因:""")
out = ["  原因        表现            排查              解决"]
out.append("  死锁        全部 rank 等      看 stack          改同步顺序")
out.append("  失配        部分 rank 不调用  对比 rank 日志    统一调用")
out.append("  网络        部分 rank 不通    ping + ibstat    修网络")
out.append("  超时        报 timeout       NCCL_DEBUG=INFO  改 NCCL_TIMEOUT")
out.append("  NCCL 版本   集群异构         check 版本        统一 NCCL")
out.append("  路由        走慢链路         NET 网卡          配 IB 网卡")
out.append("  防火墙      端口被挡         telnet 端口       关防火墙 / 开端口")
res("\n".join(out))
mea("""3 大原因比例 (经验):
  - 死锁 (代码 bug): 50%
  - 失配 (调用不一致): 25%
  - 网络/配置: 25%
排查优先级:
  1. 日志 (NCCL_DEBUG=INFO)
  2. 死锁检测 (torch.distributed.monitor)
  3. 网络测试 (ibstat, ibv_devinfo, ping)
  4. 防火墙 (telnet 22xxx)
  5. NCCL 版本 (集群统一)""")

# --- 2. 死锁场景与解决 ---
hdr(2,TOTAL,"死锁场景与解决")
why("""最常见的死锁:""")
out = ["  死锁场景             描述                  解决"]
out.append("  A 发 -> B 收, B 发 -> A 收  对称 P2P 死锁  奇偶法, 一边 send 一边 recv")
out.append("  A allreduce, B 不调        失配            统一调用")
out.append("  A barrier, B 不在          B 永远不加入     补齐 B")
out.append("  A 等 recv, B 已退出        B 没数据发      A 配 timeout")
out.append("  异 rank 调不同 op          错位            代码评审")
out.append("  嵌套 group 错              父子 group 死锁 拆 group")
out.append("  多次 send, 1 次 recv       buffer 错位     配对发送接收")
res("\n".join(out))
mea("""死锁 5 大反模式:
  1. 同步 send/recv 对称 (用奇偶)
  2. 不同 rank 调不同 op
  3. 缺 rank (有 rank 没启动)
  4. group 配置错
  5. 多 send / 1 recv (buffer 错位)
防死锁:
  1. 用 isend/irecv (非阻塞)
  2. 用 ncclGroupStart 一次提交
  3. 配 NCCL_TIMEOUT, 避免无限等
  4. 单元测试: 跑 1 次 8 rank 验证
  5. 写代码: 所有 rank 调同样的 op""")

# --- 3. NCCL_DEBUG 日志 ---
hdr(3,TOTAL,"NCCL_DEBUG 日志解读")
why("""NCCL 提供 4 级日志:""")
out = ["  级别      输出内容                              何时用"]
out.append("  WARN      只报错                              日常, 性能最佳")
out.append("  INFO      初始化 + 算法选择                    查问题")
out.append("  TRACE     每次 send/recv                       深度调试")
out.append("  ALL       全部, 极慢                          死锁定位")
out.append("  配合 SUBCOLL/COLL/P2P/...      子系统日志     精细")
res("\n".join(out))
mea("""实战日志解读:
  # INFO 日志关键行:
  NCCL INFO Channel 00 : 0[0] -> 8[0] [send] via NET/IB/0
  NCCL INFO Ring 00 : 0 -> 8 via ...   # ring 拓扑
  NCCL INFO algorithm Ring ...         # 算法选 ring
  NCCL INFO Connected all rings        # 连接成功

  # hang 时日志会卡在:
  NCCL INFO Channel 00 : 0[0] -> 4[0] [send] via ...
  # 然后一直不动 -> 4 不在, 或 4 已 hang

  # 死锁定位:
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ALL
  # 跑挂, 看哪 rank 卡在等什么

  # 关键: 各 rank 的 stack 截图
  py-spy dump --pid <pid>
  # 看 GIL 卡哪了""")

# --- 4. 排查工具与命令 ---
hdr(4,TOTAL,"排查工具与命令")
why("""实战排查命令:""")
out = ["  步骤         命令                                            作用"]
out.append("  1. 看进程     ps -ef | grep python | grep train             谁在跑")
out.append("  2. 看 GPU    nvidia-smi                                      卡占用")
out.append("  3. 看 stack   py-spy dump --pid <pid>                       python 栈")
out.append("  4. 看 GIL     kill -SIGUSR1 <pid>                           dump 线程")
out.append("  5. 网络       ibstatus                                       IB 状态")
out.append("  6. 端口       netstat -an | grep 22xxx                       端口通")
out.append("  7. 防火墙    iptables -L -n                                 防火墙规则")
out.append("  8. DNS        nslookup <master_addr>                        主机名解析")
out.append("  9. NCCL 日志  NCCL_DEBUG=INFO + 重跑                        看算法")
out.append("  10. timeout   NCCL_TIMEOUT=300 (秒)                         超时")
res("\n".join(out))
mea("""排查流程 (实战):
  1. 进程活? nvidia-smi 看 GPU 利用率
     - 0% 都在 hang
     - 1 个 0% 1 个 100%, 是该 rank 的问题
  2. 抓 stack: py-spy dump --pid <pid>
     - 卡在 dist.barrier: 失配
     - 卡在 dist.all_reduce: 死锁 / 网络
  3. 看 NCCL 日志: 哪个 rank 报 timeout
  4. 网络: ibstatus, 看 IB link up
  5. 端口: netstat 看 22xxx 通不通
  6. 防火墙: 节点间 iptables -L
  7. 版本: 集群 NCCL 版本统一 (nccl -V)
  8. 最后: 找 NCCL expert, 提 issue
预防:
  - CI 跑 1 次 8 rank 验证
  - 监控 + 报警
  - 统一 NCCL / torch / cuda 版本""")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:集合通信 hang = 卡死, 3 大原因: 死锁 (50%) / 失配 (25%) / 网络 (25%);
  排查: NCCL_DEBUG=INFO 看日志, py-spy 抓 stack, ibstatus 看网络;
  解决: 改同步顺序 (奇偶法) + 统一调用 + 修网络;
  配 NCCL_TIMEOUT 避免无限等, 单元测试验证。
- 熟手:死锁 5 大反模式: 对称 P2P / 不同 op / 缺 rank / group 错 / buffer 错位;
  NCCL 日志 4 级: WARN/INFO/TRACE/ALL, INFO 查问题;
  排查流程: nvidia-smi -> py-spy -> NCCL 日志 -> ibstatus -> 端口 -> 防火墙 -> 版本;
  防挂: CI 跑 8 rank 验证, 监控报警, 统一 NCCL/torch/cuda 版本;
  Ascend 同: HCCL 日志 + npu-smi + 端口检查。
【进阶】写 1 个能复现 hang 的最小 demo (2 rank 死锁), 实战排查:
  1. 配 NCCL_DEBUG=INFO 重跑, 截图 hang 时日志;
  2. py-spy dump 抓 stack, 定位卡哪;
  3. 改代码 (奇偶法), 验证修复;
  4. 写 1 个 CI 集成测试, 防回归。
EOF
echo "############################################################"
