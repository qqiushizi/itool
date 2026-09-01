#!/bin/bash
# ============================================================
# 实验: d.evaluation-metrics
# 说明: 准确率/精确召回/F1/AUC/混淆矩阵
# 模块: p.practise/a.ai-fundamentals  AI 基础
# ============================================================
# 【第一性原理】
# 评估指标告诉你"模型到底好不好",但不同指标看重的角度不同:
#  准确率=对的比例,类别不平衡时会骗人(全猜多数类也很高);
#  精确率=预测为正里真为正的比例(不误报);召回率=真为正里被找出来的比例(不漏报);
#  F1=精确与召回的调和平均,平衡两者;AUC=排序能力(正例分数排在前面的概率),与阈值无关。
# 本实验从一组预测概率出发,算混淆矩阵与各指标,并看阈值如何移动精确率/召回率。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 评估指标 / 混淆矩阵 / P-R-F1 / AUC"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=4, suppress=True)
# 模型预测概率 vs 真实标签(含类别不平衡:正例少)
probs=np.array([0.90,0.80,0.70,0.60,0.55,0.45,0.30,0.20,0.10,0.05])
labels=np.array([1,1,1,0,1,0,0,0,0,0])
def cm(p,t=0.5):
    pred=(p>=t).astype(int)
    TP=int(((pred==1)&(labels==1)).sum()); FP=int(((pred==1)&(labels==0)).sum())
    FN=int(((pred==0)&(labels==1)).sum()); TN=int(((pred==0)&(labels==0)).sum())
    return TP,FP,FN,TN
print("【1】阈值=0.5 时的混淆矩阵:")
TP,FP,FN,TN=cm(probs)
print(f"  预测正  预测负\n  真实正  TP={TP}    FN={FN}\n  真实负  FP={FP}    TN={TN}")
acc=(TP+TN)/len(labels); prec=TP/(TP+FP); rec=TP/(TP+FN); f1=2*prec*rec/(prec+rec)
print(f"  准确率={acc:.3f}  精确率={prec:.3f}  召回率={rec:.3f}  F1={f1:.3f}")
print("  解读:准确率0.8看着不错,但4个正例里只找对3个(召回0.75);精确率=预测为正里真为正的比例。")

print("\n【2】不同阈值下精确率/召回率的此消彼长:")
for t in [0.30,0.50,0.65,0.80]:
    TP,FP,FN,TN=cm(probs,t); p=TP/(TP+FP) if TP+FP>0 else 0; r=TP/(TP+FN) if TP+FN>0 else 0
    print(f"  阈值={t:.2f}: 精确率={p:.3f}  召回率={r:.3f}")
print("  解读:阈值降→召回升(多抓,但易误报,精确降);阈值升→精确升(少抓,但易漏,召回降)。→ P-R 是一对权衡。")

# AUC:Mann-Whitney U = 正例分数高于负例分数的概率
pos=probs[labels==1]; neg=probs[labels==0]
auc=np.mean([(p>ng) for p in pos for ng in neg])
print(f"\n【3】AUC(排序能力,与阈值无关)={auc:.3f}")
print("  解读:AUC=正例分数高于负例分数的概率=0.90(接近1说明排序好)。")
print("        AUC 不依赖阈值,衡量的是'把正例排在前面'的能力,类别不平衡时比准确率更可靠。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:准确率在不平衡时会骗人;精确率防误报、召回率防漏报,F1 平衡两者;AUC 看排序能力与阈值无关。
- 熟手:类别极不平衡看 AUC/PR-AUC 而非准确率;多分类用宏/微平均 F1 或混淆矩阵;
  指标选择要绑定业务目标(医疗重召回、垃圾邮件重精确)。
- 延伸:把正例数减到2个看准确率为何仍高;画出完整 P-R 曲线找 F1 最大点。
EOF
echo "============================================================"
