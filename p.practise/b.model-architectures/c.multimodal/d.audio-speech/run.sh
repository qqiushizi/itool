#!/bin/bash
# ============================================================
# 实验: d.audio-speech
# 说明: Mel 特征、语音识别/合成架构概览
# 模块: p.practise/b.model-architectures  模型架构
# ============================================================
# 【第一性原理】
# 语音是一维波形,直接喂神经网络效率低。先把波形分帧→加窗→做 FFT 得频谱,再用 Mel 滤波器组
# 把频率压到人耳敏感的 Mel 尺度(对数-like),得到 Mel 频谱图——语音的"图像化"表示。
# 识别(ASR):Mel 频谱→编码器(如 Conformer/Whisper)→文字;合成(TTS):文字→解码器→波形。
# 本实验从一段合成波形出发,算 FFT 频谱与 Mel 滤波器,看 Mel 尺度如何压缩频率。
# ============================================================
set -euo pipefail
if ! python3 -c "import numpy" 2>/dev/null; then
  echo "未检测到 numpy,正在安装..."; python3 -m pip install --quiet numpy || { echo "请手动: pip install numpy"; exit 1; }
fi
echo "============================================================"
echo " 实验: 语音 / Mel 特征 / ASR·TTS 架构"
echo "============================================================"
python3 <<'PY'
import numpy as np
np.set_printoptions(precision=3, suppress=True)
rng=np.random.default_rng(0)
sr=8000; t=np.linspace(0,0.5,int(sr*0.5))
# 合成语音:基频 + 谐波(模拟元音)
wave=0.5*np.sin(2*np.pi*200*t)+0.3*np.sin(2*np.pi*400*t)+0.2*np.sin(2*np.pi*600*t)
# 1 分帧 + FFT 频谱
nfft=512; frame=wave[:nfft]*np.hanning(nfft)
spec=np.abs(np.fft.rfft(frame))[:nfft//2]
freqs=np.fft.rfftfreq(nfft,1/sr)[:nfft//2]
print("【1】分帧+FFT:一段语音波形 → 频谱(能量随频率分布)")
peak=freqs[np.argmax(spec)]
print(f"  帧长 {nfft}, 频谱峰值频率≈{peak:.0f}Hz(对应基频/谐波)")
print(f"  低频段能量(0-1000Hz)={spec[freqs<1000].sum():.1f}  高频段(>1000Hz)={spec[freqs>=1000].sum():.1f}")
print("  解读:语音能量集中在低频谐波;FFT 把时域波形变成频域能量分布,更适合模型处理。")

# 2 Mel 滤波器组:对数压缩频率
def hz2mel(f): return 2595*np.log10(1+f/700)
def mel2hz(m): return 700*(10**(m/2595)-1)
nmel=10
mel_pts=np.linspace(hz2mel(0),hz2mel(sr//2),nmel+2)
hz_pts=mel2hz(mel_pts)
print(f"\n【2】Mel 尺度:把 Hz 映射到 Mel(人耳对低频敏感、高频迟钝)")
print(f"  Mel 滤波器中心频率(Hz)={np.round(hz_pts,0).tolist()}")
print("  解读:Mel 在低频分辨率高、高频压缩——和人耳听觉一致,所以用 Mel 比线性 Hz 更省维度、更有效。")

# 3 ASR / TTS 架构
print("\n【3】ASR / TTS 架构概览:")
print("  ASR(语音→文字):波形→Mel频谱图→编码器(Conformer/Whisper Transformer)→解码器→文字")
print("  TTS(文字→语音):文字→编码器→时长/声学预测→vocoder(声码器)→波形")
print("  解读:Mel 频谱图是语音任务的通用输入(像图像之于视觉);Transformer/Conformer 是主流骨干。")
PY
echo "============================================================"
cat <<'EOF'
【结论与进阶】
- 小白:语音先分帧做 FFT 得频谱,再用 Mel 滤波器按人耳敏感度压缩成 Mel 频谱图;ASR 从 Mel 转文字,TTS 反过来。
- 熟手:Mel 尺度是对数式,低频细高频粗;MFCC 还做了 DCT 去相关,老式 ASR 常用;
  Whisper 用 Mel+Transformer 实现强零样本 ASR;vocoder(HiFi-GAN)把 Mel 还原高保真波形。
- 延伸:改基频/谐波听音色变化;对比线性 Hz 与 Mel 滤波器的维度数差异。
EOF
echo "============================================================"
