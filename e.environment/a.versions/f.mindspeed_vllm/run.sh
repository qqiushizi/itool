#!/bin/bash
echo "========== MindSpeed / vLLM-Ascend Info =========="
echo ""

echo "[MindSpeed-LLM]"
python3 -c "import mindspeed; print(f'MindSpeed-LLM: {mindspeed.__version__}')" 2>/dev/null || echo "MindSpeed-LLM not installed"
pip3 list 2>/dev/null | grep -iE "mindspeed-llm|mindspeed_llm"

echo ""
echo "[MindSpeed-MM]"
python3 -c "import mindspeed_mm; print(f'MindSpeed-MM: {mindspeed_mm.__version__}')" 2>/dev/null || echo "MindSpeed-MM not installed"
pip3 list 2>/dev/null | grep -iE "mindspeed-mm|mindspeed_mm"

echo ""
echo "[MindSpeed-RL]"
python3 -c "import mindspeed_rl; print(f'MindSpeed-RL: {mindspeed_rl.__version__}')" 2>/dev/null || echo "MindSpeed-RL not installed"
pip3 list 2>/dev/null | grep -iE "mindspeed-rl|mindspeed_rl"

echo ""
echo "[vLLM-Ascend]"
python3 -c "import vllm; print(f'VLLM: {vllm.__version__}')" 2>/dev/null || echo "VLLM not installed"
pip3 list 2>/dev/null | grep -iE "vllm|ascend.*vllm"
which vllm 2>/dev/null || echo "vllm CLI not in PATH"

echo ""
echo "[vLLM Backend]"
python3 -c "import vllm; print(f'Engine args: {vllm.utils.get_engine_version() if hasattr(vllm.utils, \"get_engine_version\") else \"N/A\"}')" 2>/dev/null

echo ""
echo "[NeuBot / Megatron-LM]"
pip3 list 2>/dev/null | grep -iE "megatron|neobot|nebulastream"

echo ""
echo "[FlagScale / FlagOpen]"
pip3 list 2>/dev/null | grep -iE "flagscale|flagopen|flagperf"

echo ""
echo "[MindIE / MindSPONGE]"
pip3 list 2>/dev/null | grep -iE "mindie|mindsp|AscendCL|ascendcl"

echo ""
echo "[CANN ACL Runtime]"
python3 -c "import acl; print(f'ACL: {acl.__version__}')" 2>/dev/null || echo "ACL not installed"
pip3 list 2>/dev/null | grep -iE "^acl|ascend.*acl"

echo ""
echo "[Relevant pip packages for Ascend]"
pip3 list 2>/dev/null | grep -iE "ascend|mindsp|mindspeed|nebot|flag"

echo ""
echo "================================================"
