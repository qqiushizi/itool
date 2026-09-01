#!/bin/bash
# ============================================================
# 实验: d.detection-segmentation
# 说明: 检测头/anchor、分割上采样直觉
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 检测:不只分类,还要"在哪、多大"→预测框+类别。anchor 是预先铺在每个网格的参考框,
# 模型预测相对 anchor 的偏移;用 IoU 判断框重合度,用 NMS 去掉重叠的重复框。
# 分割:逐像素分类,需要把低分辨率特征图上采样回原图分辨率(转置卷积/双线性)。
# 本实验手算 IoU、NMS,并演示上采样如何还原分辨率。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 检测 / anchor / IoU / NMS / 分割上采样"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih
    area_a=(ax2-ax1)*(ay2-ay1); area_b=(bx2-bx1)*(by2-by1)
    return inter/(area_a+area_b-inter+1e-12)

# 1 IoU
gt=[10,10,30,30]
preds={"好框":[12,12,32,32],"偏框":[20,20,40,40]}
print("【1】IoU(交并比):预测框与真值框的重合度")
print(f"  真值框={gt}")
for n,b in preds.items():
    print(f"  {n}{b} → IoU={iou(gt,b):.3f}")
print("  解读:IoU>0.5 通常算检测正确;IoU 是评估和 NMS 的基础。")

# 2 NMS
print("\n【2】NMS:去掉重叠的重复检测框(保留分数最高的)")
boxes=np.array([[10,10,30,30],[11,11,31,31],[50,50,70,70]],float)
scores=np.array([0.9,0.8,0.7])
order=scores.argsort()[::-1]; keep=[]
while len(order):
    i=int(order[0]); keep.append(i)
    order=[j for j in order[1:] if iou(boxes[i],boxes[j])<0.5]
print(f"  候选框(含分数)={list(zip(boxes.tolist(),scores.tolist()))}")
print(f"  NMS 后保留框索引={keep}  (前两框高度重叠,保留分数高的0.9)")
print("  解读:同一物体常被多个框检出,NMS 按分数排序,删掉与高分框 IoU 过大的低分框。")

# 3 anchor
print("\n【3】anchor:预铺参考框,模型只预测相对偏移")
grid=(4,4); anchors_per=3
print(f"  {grid[0]}×{grid[1]} 网格 × {anchors_per} 种尺度/比例 = {grid[0]*grid[1]*anchors_per} 个 anchor")
print("  解读:anchor 密集铺满图,模型只需预测每个 anchor 的类别+小幅偏移,把'从零生成框'变成'微调预设框'。")

# 4 分割上采样
print("\n【4】分割上采样:把低分辨率特征图还原到原图分辨率")
low=np.array([[0,1],[2,3]],float)
up=np.kron(low,np.ones((2,2)))   # 最近邻 2× 上采样
print(f"  低分辨率 {low.shape}:\n{low}\n  上采样 2× {up.shape}:\n{up.astype(int)}")
print("  解读:分割要逐像素标签,但骨干不断下采样;上采样(转置卷积/双线性)把分辨率还原回原图,再逐像素分类。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:检测预测框+类别,用 IoU 量重合、NMS 去重复、anchor 当参考框;分割逐像素分类,靠上采样还原分辨率。
- 熟手:anchor-free(YOLOX/FCOS)也在兴起,免去预设框;多尺度检测用 FPN 在不同层检测不同大小目标;
  分割上采样可用转置卷积(可学)或双线性(固定);语义分割 vs 实例分割 vs 全景分割层次不同。
- 延伸:把 NMS 的 IoU 阈值从0.5调到0.7看保留框变化;改 anchor 尺度比例看覆盖。
EOF
echo "============================================================"
