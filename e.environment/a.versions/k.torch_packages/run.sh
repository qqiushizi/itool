#!/bin/bash
echo "========== PyTorch & NPU Packages Info =========="
echo ""

echo "[Python Version]"
python3 --version 2>/dev/null || python --version 2>/dev/null

echo ""
echo "[Installed PyTorch Packages]"
pip3 list 2>/dev/null | grep -iE "torch|npu|ascend" || pip list 2>/dev/null | grep -iE "torch|npu|ascend"

echo ""
echo "[Torch Version]"
python3 -c "import torch; print(f'Torch: {torch.__version__}')" 2>/dev/null || echo "Torch not installed"

echo ""
echo "[Torch CUDA/NPU Availability]"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'NPU available: {torch.npu.is_available() if hasattr(torch, \"npu\") else \"N/A\"}')" 2>/dev/null || echo "Cannot check"

echo ""
echo "[Torchvision Version]"
python3 -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')" 2>/dev/null || echo "Torchvision not installed"

echo ""
echo "[Torchaudio Version]"
python3 -c "import torchaudio; print(f'Torchaudio: {torchaudio.__version__}')" 2>/dev/null || echo "Torchaudio not installed"

echo ""
echo "[Torch NPU Version]"
python3 -c "import torch_npu; print(f'Torch_NPU: {torch_npu.__version__}')" 2>/dev/null || echo "Torch_NPU not installed"

echo ""
echo "[Torch NPU Build Info]"
python3 -c "import torch_npu; print(f'NPU backend: {torch_npu.npu.get_npu_backend_type()}')" 2>/dev/null || echo "NPU backend info not available"

echo ""
echo "[Check torch_npu.sys_config]"
python3 -c "import torch_npu.sys_config as sys_config; print(sys_config.show())" 2>/dev/null || echo "sys_config not available"

echo ""
echo "[Relevant Environment Variables]"
echo "LD_LIBRARY_PATH: $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -iE 'torch|npu|ascend|cann' | head -5)"
echo "PYTORCH_TEST_CUDA: $PYTORCH_TEST_CUDA"
echo "NPU_ACCEL_VERSION: $NPU_ACCEL_VERSION"

echo ""
echo "[Installed torch-* packages via pip]"
pip3 list 2>/dev/null | grep "^torch"

echo ""
echo "================================================="
