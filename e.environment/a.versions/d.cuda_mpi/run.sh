#!/bin/bash
echo "========== CUDA / OpenMPI / NCCL Info =========="
echo ""

echo "[CUDA Version]"
nvcc --version 2>/dev/null || echo "nvcc not found"
cat /usr/local/cuda/version.txt 2>/dev/null || cat /usr/local/cuda/version.json 2>/dev/null || echo "CUDA version file not found"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"

echo ""
echo "[NCCL Version]"
python3 -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')" 2>/dev/null || echo "NCCL via torch not available"
which nccl-info 2>/dev/null || echo "nccl-info not in PATH"
ls /usr/local/cuda/nccl* 2>/dev/null || echo "NCCL not found in CUDA path"

echo ""
echo "[OpenMPI Version]"
mpirun --version 2>/dev/null || ompi-info --version 2>/dev/null || echo "OpenMPI not found"
which mpirun ompi-info 2>/dev/null

echo ""
echo "[NCCL from pip]"
pip3 list 2>/dev/null | grep -i nccl

echo ""
echo "[OpenMPI / MPI from pip]"
pip3 list 2>/dev/null | grep -iE "mpi|openmpi|mpich"

echo ""
echo "================================================"
