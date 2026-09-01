#!/bin/bash
# ============================================================
# 实验: a.cnn-basics
# 说明: 卷积/池化/感受野、特征图与参数量
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 卷积:一个小核(如3×3)滑过图像,每处做点积→一个特征图。关键优势:权重共享(同一核扫全图)
# 和局部连接(只看邻域),所以参数远少于全连接,且对平移有不变性。
# 池化:下采样(取最大/平均),降分辨率、扩大感受野、带来轻微平移不变性。
# 感受野:一个输出像素能"看到"原图多大区域,层数越多越大。本实验手算卷积、池化、感受野与参数量。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 卷积 / 池化 / 感受野 / 特征图 / 参数量"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=2, suppress=True)
def conv2d(img,k):
    kh,kw=k.shape; oh,ow=img.shape[0]-kh+1,img.shape[1]-kw+1
    out=np.zeros((oh,ow))
    for i in range(oh):
        for j in range(ow):
            out[i,j]=(img[i:i+kh,j:j+kw]*k).sum()
    return out
def maxpool2(x):
    h=x.shape[0]//2; out=np.zeros((h,h))
    for i in range(h):
        for j in range(h):
            out[i,j]=x[2*i:2*i+2,2*j:2*j+2].max()
    return out
# 1 卷积
img=np.array([[1,2,0,1,3,2],[0,1,3,2,1,0],[2,0,1,2,0,3],[1,3,2,0,1,2],[0,1,2,3,1,0],[2,1,0,2,3,1]],float)
k=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],float)   # 垂直边缘检测核
feat=conv2d(img,k)
print("【1】卷积:3×3 核滑过 5×5 图 → 特征图")
print(f"  输入图 {img.shape}:\n{img}")
print(f"  卷积核(垂直边缘):\n{k}")
print(f"  特征图 {feat.shape}:\n{feat}")
print("  解读:核在响应'垂直边缘'的位置输出大;同一组权重扫全图=参数共享,远比全连接省参数。")

# 2 池化
p=maxpool2(feat)
print(f"\n【2】最大池化 2×2:特征图 {feat.shape} → {p.shape}\n{p}")
print("  解读:取每个2×2块的最大值→分辨率减半、保留最强响应、感受野翻倍,带来轻微平移不变性。")

# 3 感受野 & 参数量
print("\n【3】感受野与参数量(堆叠 3×3 卷积层):")
rf=1
for L in range(1,5):
    rf+=2                    # 每层 3×3 卷积感受野+2
    print(f"  {L} 层 3×3 卷积: 感受野={rf}×{rf}")
cin,cout=3,64
params_conv=3*3*cin*cout     # 一个卷积层参数
params_fc=224*224*cin*1024   # 若用全连接把 224×224×3 接到 1024
print(f"  一个 3×3 卷积(Cin={cin}→Cout={cout})参数 = {params_conv}")
print(f"  对比全连接(224×224×3→1024)参数 = {params_fc:,}  (相差 {params_fc//params_conv} 倍)")
print("  解读:卷积靠权重共享+局部连接,参数量与输入分辨率无关;全连接参数随分辨率平方爆炸。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:卷积用小核扫全图提特征(权重共享、省参数、抗平移);池化降分辨率扩感受野;
  层数越多感受野越大,能看到的原图区域越大。
- 熟手:感受野要覆盖目标尺度才有效;现代设计常用 stride/空洞卷积快速扩感受野;
  参数量=kh×kw×Cin×Cout(+bias),与空间尺寸无关;特征图尺寸=(H-k+1)无padding。
- 延伸:加 padding=1 看输出尺寸是否保持;换成 1×1 卷积看它只做通道混合;算 VGG16 的总参数量。
EOF
echo "============================================================"
