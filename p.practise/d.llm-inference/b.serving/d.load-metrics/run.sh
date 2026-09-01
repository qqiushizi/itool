#!/bin/bash
# ============================================================
# 实验: d.load-metrics
# 说明: 吞吐/延迟/并发、SLO 概念
# 模块: p.practise 第一性原理学习实验
# ============================================================
# 【第一性原理】
# LLM 服务的核心指标:
#   - 吞吐: tokens/s 或 requests/s
#   - 延迟: TTFT (首字) + TPOT (每字)
#   - 并发: 同时服务的请求数
# SLO (Service Level Objective):
#   - TTFT P99 < 500ms
#   - TPOT P99 < 50ms
#   - 吞吐 > X tokens/s
# 三大指标的取舍:
#   - 想要高吞吐 -> 加大 batch, 但 TPOT 会涨
#   - 想要低延迟 -> 小 batch, 但吞吐低
#   - 想要多并发 -> 优化显存, 但单请求慢
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then python3 -m pip install --quiet numpy; fi
echo "############################################################"
echo "# 实验: d.load-metrics | 吞吐/延迟/并发:SLO 设定与权衡"
echo "############################################################"

python3 <<'PYEOF'
import numpy as np
def hdr(n,t,total): print(f"\n{'='*60}\n【步骤 {n}/{total}】{t}\n{'='*60}")
def why(s): print("\n--- 做什么 & 为什么 ---\n  "+s.replace("\n","\n  "))
def res(s): print("\n--- 结果 ---\n  "+s.replace("\n","\n  "))
def mea(s): print("\n--- 解读 ---\n  "+s.replace("\n","\n  "))
TOTAL = 4

# --- 1. 核心指标定义 ---
hdr(1,TOTAL,"LLM 服务 5 大核心指标")
why("生产环境必看的 5 个数:")
res("""指标                  定义                        用户感受
  TTFT                首字时间 (P50/P99)         等多久开始出字
  TPOT                后续每字时间 (P50/P99)     流式顺不顺
  吞吐 (tokens/s)     总生成 token 数/秒         整体能服务多少
  吞吐 (req/s)        总请求数/秒                业务并发能力
  并发数 (active req) 同时在跑的请求数             服务承载力""")
mea("P50/P99/P999 是分位数。P99 = 99% 请求比这快,1% 比这慢。\n  设 SLO 用 P99 才有意义——只看平均会被长尾拖死。")

# --- 2. 吞吐 vs 延迟:经验曲线 ---
hdr(2,TOTAL,"batch 大小 vs 吞吐/延迟")
why("大模型推理 batch 越大,GPU 利用率越高,吞吐越高,但单请求 TPOT 涨。\n  经验:batch 翻倍,吞吐 ~1.5x,TPOT ~1.5-2x")
bss = [1, 4, 8, 16, 32, 64, 128]
out = ["  batch   吞吐 tok/s   TPOT(ms)   TTFT(ms)", "  " + "-"*50]
for bs in bss:
    tput = 50 + 40*bs
    tpot = 20 + 0.3*bs
    ttft = 50 + 1.5*bs
    out.append(f"  {bs:5d}    {tput:5d}        {tpot:6.1f}     {ttft:6.1f}")
res("\n".join(out))
mea("batch=64 是甜点:高吞吐,TPOT 仍 < 50ms。\n  batch=128: 吞吐再涨 30%,但 TPOT 接近 60ms,逼近 SLO 上限。\n  生产调优:在 SLO 不违约前提下,选最大 batch。")

# --- 3. SLO 设定 ---
hdr(3,TOTAL,"SLO 设定模板(按业务场景)")
why("不同场景 SLO 不同:")
res("""场景              TTFT P99     TPOT P99     吞吐
  实时聊天           < 300ms      < 50ms       ~500 tok/s
  搜索(Query)        < 200ms      < 30ms       ~2000 req/s
  离线批处理         < 5s         < 100ms      ~5000 tok/s
  代码补全           < 100ms      < 30ms       ~3000 req/s
  长文档总结(慢)      < 2s         < 80ms       ~200 req/s""")
mea("SLO 是承诺给业务的线。超出要赔钱/丢客户;过松 = 浪费算力。\n  关键: SLO 用 P99 设,但内部监控用 P50/P95/P99/P99.9 多维看。")

# --- 4. 监控 + 报警 ---
hdr(4,TOTAL,"监控维度 + 报警规则")
why("生产必监控:")
res("""监控项                  告警阈值          排查
  GPU 利用率            < 50% 持续 1min   调度问题/batch 小
  KV cache 利用率       > 90%             接近 OOM,扩 batch 上限
  TTFT P99              > SLO * 1.2      prefill 慢/chunked 未开
  TPOT P99              > SLO * 1.2      decode 慢/KV 大
  队列等待时间          > 5s              请求堆积,扩并发
  OOM 次数              > 0               配置问题
  错误率 (4xx/5xx)      > 0.5%            业务/参数问题
  显存                  > 95%             接近 OOM""")
mea("工具:Grafana + Prometheus + vllm metrics endpoint。\n  vllm 暴露 /metrics(Prometheus 格式),含 ttft/tpot/gpu_cache 等。\n  高级:用 shadow traffic 测新模型/新配置,不影响线上。")
PYEOF

echo ""
echo "############################################################"
cat <<'EOF'
【整体总结】
- 小白:核心指标是 TTFT(首字)+ TPOT(每字)+ 吞吐;SLO 用 P99 设;
  batch 越大吞吐高但 TPOT 涨,要在 SLO 不违约下选最大 batch。
- 熟手:监控 /metrics(Prometheus)+ 告警 + shadow test 是上线标配;
  TTFT P99 / TPOT P99 / KV cache 利用率 / GPU 利用率是黄金 4 件套;
  业务分级 SLO(聊天 vs 离线)避免一刀切。
【进阶】用 Locust / vegeta 压测,生成 P50/P99 延迟报告;用 Grafana 画 7 天趋势。
EOF
echo "############################################################"
