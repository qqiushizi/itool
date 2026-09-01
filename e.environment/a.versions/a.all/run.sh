#!/bin/bash
# Collect version info for all Ascend / MindSpore related software in one go.
# Items: hdk (driver/npu-smi), cann, torch / torch_npu, transformers,
# vllm-ascend, triton-ascend, plus common training/inference deps.

set +e

section() { echo ""; echo "===== $1 ====="; }
have()     { command -v "$1" >/dev/null 2>&1; }
pyver()    { python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null; }
pipver()   { python3 -m pip --version 2>/dev/null | awk '{print $1,$2}'; }
pkgver()   {
    # pkgver <python-module> [pip-name]
    local mod="$1" name="${2:-$1}"
    python3 -c "import importlib,sys; m=importlib.import_module('$mod'); print(getattr(m,'__version__','?'))" 2>/dev/null \
        || python3 -m pip show "$name" 2>/dev/null | awk '/^Version:/{print $2}'
}

section "Host / OS"
echo "Hostname   : $(hostname 2>/dev/null)"
echo "Kernel     : $(uname -a)"
( . /etc/os-release 2>/dev/null && echo "Distro     : $PRETTY_NAME ($ID $VERSION_ID)" ) || echo "Distro     : (unknown)"
echo "Python     : $(which python3) ($(pyver))"
echo "Pip        : $(pipver)"
echo "GLIBC      : $(ldd --version 2>/dev/null | head -1)"
echo "GCC        : $(gcc --version 2>/dev/null | head -1)"
echo "CUDA       : $(nvcc --version 2>/dev/null | tail -1)"

section "Ascend HDK (Driver / NPU)"
if have npu-smi; then
    npu-smi info 2>/dev/null
else
    echo "npu-smi not found"
fi
if have npuinfo; then
    npuinfo 2>/dev/null | head -40
fi
echo ""
echo "[Driver install info]"
cat /etc/ascend_install.info 2>/dev/null || echo "(no /etc/ascend_install.info)"
echo ""
echo "[lspci NPU]"
lspci 2>/dev/null | grep -i ascend || echo "(no ascend device in lspci)"
echo ""
echo "[/dev devices]"
ls -la /dev/davinci* /dev/ascend* /dev/hisi* 2>/dev/null || echo "(no ascend device files)"
echo ""
echo "[Driver / firmware packages]"
{ dpkg -l 2>/dev/null | awk '/^ii/ {print $2}' | grep -E '^(Ascend-|Huawei-)(hdk|driver|npu|firmware|toolkit)'; \
  rpm -qa 2>/dev/null | grep -E '^(Ascend-|Huawei-)(hdk|driver|npu|firmware|toolkit)'; } | sort -u
[ $? -ne 0 ] && true
# fallback: look in versioned dirs
ls -d /usr/local/Ascend/Ascend-* 2>/dev/null || true

section "CANN"
for v in ASCEND_HOME ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH \
         ASCEND_RUNTIME_HOME ASCEND_NNAE_HOME ASCEND_AOE_HOME; do
    val="$(eval echo \$$v)"
    if [ -n "$val" ]; then echo "$v = $val"; fi
done
echo ""
echo "[CANN install root + version files]"
for path in "$ASCEND_HOME" "$ASCEND_TOOLKIT_HOME" "/usr/local/Ascend" "/home/ascend"; do
    if [ -d "$path" ]; then
        # scan one level for toolkits like cann-X.Y.Z or Ascend-Driver
        for d in "$path" "$path"/*/; do
            [ -d "$d" ] || continue
            for vf in version.cfg version version.info; do
                if [ -f "$d/$vf" ]; then
                    echo "--- $d/$vf ---"
                    cat "$d/$vf"
                fi
            done
            if [ -f "$d/opp/version.info" ]; then
                echo "--- $d/opp/version.info ---"
                cat "$d/opp/version.info"
            fi
        done
    fi
done
echo ""
echo "[CANN pip packages]"
python3 -m pip list 2>/dev/null | grep -Ei 'hccl|aicpu|te|topo|cann|opc|opbuild|msprof|sglang|atb|llm-datadist|llm-serving' || echo "(none)"

section "Torch / torch_npu"
echo "torch        : $(pkgver torch torch)"
echo "torch_npu    : $(pkgver torch_npu torch_npu)"
python3 - <<'PY' 2>/dev/null
try:
    import torch, torch_npu
    print(f"torch        : {torch.__version__} (built with CUDA {torch.version.cuda})")
    print(f"torch_npu    : {torch_npu.__version__}")
    if hasattr(torch_npu, "npu"):
        devs = torch_npu.npu.device_count()
        print(f"  npu devices: {devs}")
        for i in range(devs):
            print(f"    [{i}] {torch_npu.npu.get_device_name(i)}  cap={torch_npu.npu.get_device_capability(i)}")
except Exception as e:
    print(f"(torch_npu probe failed: {e})")
PY

section "Transformers / Accelerate / DeepSpeed"
echo "transformers : $(pkgver transformers transformers)"
echo "accelerate   : $(pkgver accelerate accelerate)"
echo "deepspeed    : $(pkgver deepspeed deepspeed)"
echo "safetensors  : $(pkgver safetensors safetensors)"
echo "tokenizers   : $(pkgver tokenizers tokenizers)"
echo "huggingface-hub: $(pkgver huggingface_hub huggingface-hub)"
echo "datasets     : $(pkgver datasets datasets)"
python3 -m pip show transformers 2>/dev/null | awk -F': ' '/^Name|^Version|^Location/{print}'

section "vLLM / vllm-ascend"
echo "vllm         : $(pkgver vllm vllm)"
echo "vllm-ascend  : $(pkgver vllm_ascend vllm-ascend)"
python3 - <<'PY' 2>/dev/null
try:
    import vllm
    print(f"vllm         : {vllm.__version__}")
except Exception as e:
    print(f"vllm import failed: {e}")
try:
    import vllm_ascend
    print(f"vllm-ascend  : {getattr(vllm_ascend,'__version__','?')}")
    print(f"  path       : {vllm_ascend.__file__}")
except Exception as e:
    print(f"vllm_ascend import failed: {e}")
PY

section "Triton / triton-ascend"
echo "triton       : $(pkgver triton triton)"
echo "triton-ascend: $(pkgver triton_ascend triton-ascend)"
python3 - <<'PY' 2>/dev/null
try:
    import triton
    print(f"triton       : {triton.__version__}")
except Exception as e:
    print(f"triton import failed: {e}")
try:
    import triton_ascend
    print(f"triton-ascend: {getattr(triton_ascend,'__version__','?')}")
    print(f"  path       : {triton_ascend.__file__}")
except Exception as e:
    print(f"triton_ascend import failed: {e}")
PY

section "MindSpore / MindSpeed / MindIE (optional)"
echo "mindspore    : $(pkgver mindspore mindspore)"
echo "mindspeed    : $(pkgver mindspeed mindspeed)"
echo "mindspeed-llm: $(pkgver mindspeed_llm mindspeed-llm)"
echo "mindie       : $(pkgver mindie mindie)"
python3 -m pip list 2>/dev/null | grep -Ei 'mindspore|mindspeed|mindie|llm-datadist|llm-serving' || echo "(none)"

section "Common scientific stack"
for m in numpy pandas scipy scikit-learn matplotlib pillow requests tqdm pyyaml; do
    v=$(pkgver "$m" "$m")
    [ -n "$v" ] && printf '  %-14s : %s\n' "$m" "$v"
done

section "ONNX / Engine"
echo "onnx         : $(pkgver onnx onnx)"
echo "onnxruntime  : $(pkgver onnxruntime onnxruntime)"
echo "onnxsim      : $(pkgver onnxsim onnxsim)"
python3 -m pip list 2>/dev/null | grep -Ei '^onnx' || true

section "Networking / MPI / NCCL"
nccl_ver=$(python3 -c "import torch; print(torch.cuda.nccl.version())" 2>/dev/null)
[ -n "$nccl_ver" ] && echo "torch NCCL    : $nccl_ver"
have mpirun && mpirun --version 2>&1 | head -1
have mpicc  && mpicc --version 2>&1 | head -1
have hccn_tool && hccn_tool -i 2>/dev/null | head -20
have hccl_tool && hccl_tool -i 2>/dev/null | head -20

section "Environment summary"
echo "PATH (ascend)         : $(echo "$PATH"       | tr ':' '\n' | grep -i ascend | head -5)"
echo "LD_LIBRARY_PATH(ascend): $(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -i ascend | head -5)"
echo "PYTHONPATH (head 5)   : $(echo "$PYTHONPATH"  | tr ':' '\n' | head -5)"

section "Done"
echo "All version checks finished at $(date '+%F %T')."
