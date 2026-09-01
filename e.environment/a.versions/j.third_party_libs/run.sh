#!/bin/bash
echo "========== Third-Party Libraries Info =========="
echo ""

echo "[Deep Learning Frameworks]"
pip3 list 2>/dev/null | grep -iE "transformers|diffusers|accelerate|peft|bitsandbytes|triton|tilelang|flash-attn|xformers" || echo "No matching packages found"

echo ""
echo "[Transformers]"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers not installed"
python3 -c "import transformers; print(f'Build config: {transformers.build_tensorflow_codebert_config() if hasattr(transformers, \"build_tensorflow_codebert_config\") else transformers.__file__}')" 2>/dev/null

echo ""
echo "[Triton]"
python3 -c "import triton; print(f'Triton: {triton.__version__}')" 2>/dev/null || echo "Triton not installed"
which triton 2>/dev/null || echo "triton CLI not in PATH"

echo ""
echo "[Triton-NPU/TILelang]"
pip3 list 2>/dev/null | grep -iE "triton|tilelang"

echo ""
echo "[Flash Attention]"
python3 -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" 2>/dev/null || echo "Flash Attention not installed"

echo ""
echo "[XFormers]"
python3 -c "import xformers; print(f'XFormers: {xformers.__version__}')" 2>/dev/null || echo "XFormers not installed"

echo ""
echo "[Accelerate]"
python3 -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null || echo "Accelerate not installed"

echo ""
echo "[PEFT]"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')" 2>/dev/null || echo "PEFT not installed"

echo ""
echo "[DeepSpeed]"
python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')" 2>/dev/null || echo "DeepSpeed not installed"

echo ""
echo "[VLLM]"
python3 -c "import vllm; print(f'VLLM: {vllm.__version__}')" 2>/dev/null || echo "VLLM not installed"

echo ""
echo "[TGI (Text Generation Inference)]"
pip3 list 2>/dev/null | grep -i "text-generation" || echo "TGI not found"

echo ""
echo "[ONNX Runtime]"
python3 -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')" 2>/dev/null || echo "ONNX Runtime not installed"

echo ""
echo "[OpenMMLab]"
pip3 list 2>/dev/null | grep -iE "mmcv|mmdet|mmseg|mmpretrain"

echo ""
echo "[HuggingFace Hub]"
python3 -c "import huggingface_hub; print(f'HuggingFace Hub: {huggingface_hub.__version__}')" 2>/dev/null || echo "HuggingFace Hub not installed"

echo ""
echo "[Safetensors]"
python3 -c "import safetensors; print(f'Safetensors: {safetensors.__version__}')" 2>/dev/null || echo "Safetensors not installed"

echo ""
echo "[SentencePiece]"
python3 -c "import sentencepiece; print(f'SentencePiece: {sentencepiece.__version__}')" 2>/dev/null || echo "SentencePiece not installed"

echo ""
echo "[protobuf]"
python3 -c "import google.protobuf; print(f'Protobuf: {google.protobuf.__version__}')" 2>/dev/null || echo "Protobuf not installed"

echo ""
echo "[numpy/pandas]"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy not installed"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')" 2>/dev/null || echo "Pandas not installed"

echo ""
echo "[scipy/sklearn]"
python3 -c "import scipy; print(f'SciPy: {scipy.__version__}')" 2>/dev/null || echo "SciPy not installed"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')" 2>/dev/null || echo "Scikit-learn not installed"

echo ""
echo "================================================"
