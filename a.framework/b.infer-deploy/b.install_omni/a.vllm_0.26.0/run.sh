# Pin vLLM and vLLM-Ascend to the v0.26 release line
mkdir -p /workspace/vllm-omni
cd /workspace/vllm-omni

git clone -b v0.26.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

git clone -b releases/v0.26.0rc https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -v -e .
cd ..

# Install vLLM-Omni from the latest main branch
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

pip install -v -e . --no-build-isolation
# or VLLM_OMNI_TARGET_DEVICE=npu pip install -v -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
