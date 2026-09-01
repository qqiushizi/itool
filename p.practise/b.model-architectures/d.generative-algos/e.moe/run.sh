#!/bin/bash
# ============================================================
# 实验: e.moe
# 说明: 专家路由、负载均衡、稀疏激活
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# MoE(混合专家):用一个大门控(router)给每个 token 选 top-k 个专家处理,其余专家不激活。
# 好处:参数总量大(容量大)但每个 token 只激活一小部分→算力近线性于激活量,性价比高。
# 难点:路由不均(都往少数专家挤)→ 加负载均衡损失,鼓励 token 均匀分到各专家。
# 本实验演示门控路由、top-k 选择、稀疏激活比例,以及负载均衡损失如何拉平专家使用率。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: MoE / 专家路由 / 稀疏激活 / 负载均衡"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)
E=4; n_tok=8; d=6; topk=2
x=rng.standard_normal((n_tok,d))
Wg=rng.standard_normal((d,E))
gate=softmax(x@Wg)             # 每个token对各专家的路由概率
# 1 路由 + top-k
print("【1】门控路由:每个 token 选 top-%d 个专家(共%d专家)"%(topk,E))
for i in range(4):
    top=np.argsort(gate[i])[::-1][:topk]
    print(f"  token{i}: 路由概率={gate[i].round(3).tolist()} → 选专家 {top.tolist()}")
print("  解读:router 给每个专家打分,只激活 top-k,其余专家对这个 token 不计算→稀疏。")

# 2 稀疏激活比例
active=topk*n_tok; total=E*n_tok
print(f"\n【2】稀疏激活:每个token激活{topk}专家 → 总激活{active}/{total} = {active/total*100:.0f}%")
print("  解读:参数总量=E×专家大小(大),但每token只算 top-k 个→算力≈激活比例,性价比高。")

# 3 负载均衡
# 无均衡:模拟路由偏好(让gate偏专家0)
biased_gate=gate.copy(); biased_gate[:,0]*=3; biased_gate=biased_gate/biased_gate.sum(1,keepdims=True)
usage_biased=(biased_gate.argmax(1)==np.arange(E)[:,None]).sum(1) if False else np.bincount(biased_gate.argmax(1),minlength=E)
usage_uniform=np.bincount(gate.argmax(1),minlength=E)
print(f"\n【3】负载均衡:各专家被选次数(按argmax)={usage_uniform.tolist()}  (理想是均匀)")
# 均衡损失:鼓励各专家被选概率相近
p_mean=biased_gate.mean(0); bal_loss=E*np.sum(p_mean*p_mean)   # 辅助损失 ∝ Σ p_i²
print(f"  路由概率的专家均值(偏置后)={p_mean.round(3).tolist()}  均衡损失={bal_loss:.3f}")
print("  解读:若都往少数专家挤,均衡损失变大;训练时最小化它,把 token 均匀摊到各专家,避免拥堵和浪费。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:MoE 用 router 给每个 token 选 top-k 专家,参数多但每 token 只激活一小部分,省算力;加负载均衡损失防拥堵。
- 熟手:稀疏 MoE 的算力≈激活量×专家大小,参数/算力解耦;负载均衡损失常用 Σ f_i·p_i(f=被选频率,p=平均概率);
  top-2 是主流;expert capacity 和 token dropping 处理负载溢出;DeepSeek 用细粒度专家+共享专家。
- 延伸:把 top-k 从2改到4看激活比例;去掉负载均衡损失看是否某专家被挤爆。
EOF
echo "============================================================"
