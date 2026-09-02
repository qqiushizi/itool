#!/bin/bash
# ============================================================
# ⑤ 进入容器检查软件包, 确认可开始算子开发
# 功能: 在指定容器内检查 CANN/torch/torch_npu/编译工具链/NPU 设备,
#       给出 GO / NO-GO 结论与下一步建议。
# 用法:
#   bash run.sh [容器名]
# 环境变量: NAME=容器名
# ============================================================
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

ask() {
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
}

command -v docker >/dev/null 2>&1 || { echo -e "${RED}未找到 docker。${RESET}" >&2; exit 1; }

NAME="${1:-${NAME:-}}"
[ -z "$NAME" ] && ask "容器名" "asc_dev" && NAME="$REPLY"

docker inspect "$NAME" >/dev/null 2>&1 || { echo -e "${RED}容器不存在: $NAME${RESET}" >&2; exit 1; }
docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || { echo -e "${RED}容器未运行: $NAME${RESET}" >&2; exit 1; }

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ⑤ 容器内软件包检查: $NAME${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"

docker exec -i "$NAME" bash -s <<'INNER'
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'
ok()   { echo -e "  ${GREEN}[ OK ]${RESET} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${RESET} $1"; }
bad()  { echo -e "  ${RED}[缺失]${RESET} $1"; }

GO=1

echo ""
echo "===== 1. 系统与 Python ====="
echo "  $(uname -m) / $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "  python3: $(python3 --version 2>&1 || echo 未找到)"
python3 -c "import sys; print('  python', sys.version.split()[0])" 2>/dev/null

echo ""
echo "===== 2. CANN 环境 ====="
SETENV=""
for s in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh /usr/local/Ascend/set_env.sh; do
    [ -f "$s" ] && SETENV="$s" && break
done
if [ -n "$SETENV" ]; then
    ok "找到 CANN 激活脚本: $SETENV"
    for vf in /usr/local/Ascend/ascend-toolkit/version.cfg /usr/local/Ascend/ascend-toolkit/latest/version.cfg; do
        [ -f "$vf" ] && { echo "  $(grep -iE '^version' "$vf" | head -1 | sed 's/^/version: /')"; break; }
    done
else
    bad "未找到 set_env.sh (镜像可能不含 CANN toolkit)"; GO=0
fi

# 在子进程 source 后测试 acl
if [ -n "$SETENV" ]; then
    if ( . "$SETENV" && python3 -c "import acl; print('acl', getattr(acl,'__version__','?'))" ) 2>/dev/null; then
        ok "acl 可导入 (CANN Python 接口正常)"
    else
        bad "acl 导入失败 (需先 source $SETENV 后再测)"; GO=0
    fi
fi

echo ""
echo "===== 3. 框架与算子库 ====="
if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then
    ok "torch 已安装"
else
    bad "torch 未安装"; GO=0
fi
if python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)" 2>/dev/null; then
    ok "torch_npu 已安装"
else
    bad "torch_npu 未安装 (算子开发必需)"; GO=0
fi
python3 -c "import pybind11; print('pybind11', pybind11.__version__)" 2>/dev/null && ok "pybind11 已安装" || warn "pybind11 未安装"
python3 -c "import vllm_ascend; print('vllm-ascend', getattr(vllm_ascend,'__version__','?'))" 2>/dev/null && ok "vllm-ascend 已安装" || warn "vllm-ascend 未安装(可后装)"

echo ""
echo "===== 4. 编译工具链 ====="
command -v gcc >/dev/null 2>&1 && ok "gcc: $(gcc --version | head -1)" || { bad "gcc 未安装"; GO=0; }
command -v g++ >/dev/null 2>&1 && ok "g++: $(g++ --version | head -1)" || bad "g++ 未安装"
command -v cmake >/dev/null 2>&1 && ok "cmake: $(cmake --version | head -1)" || bad "cmake 未安装"

echo ""
echo "===== 5. NPU 设备(容器内) ====="
if [ -n "$SETENV" ]; then
    ( . "$SETENV" && python3 - <<'PY' 2>/dev/null
try:
    import torch, torch_npu
    n = torch_npu.npu.device_count()
    print("  NPU 设备数:", n)
    for i in range(n):
        print("    [%d] %s" % (i, torch_npu.npu.get_device_name(i)))
except Exception as e:
    print("  (torch_npu 探测失败:", e, ")")
PY
    )
else
    if ls /dev/davinci* 1>/dev/null 2>&1; then
        ls /dev/davinci* 2>/dev/null | sed 's/^/  /'
    else
        echo "  (无 /dev/davinci*)"
    fi
fi

echo ""
echo "===== 6. 结论 ====="
if [ "$GO" = "1" ]; then
    echo -e "  ${GREEN}✅ GO: 容器内软件包齐全, 可以开始算子开发。${RESET}"
    echo "  建议下一步:"
    echo "    · 算子需求分析:  bash d.ops_develop/c.design/a.op_spec/run.sh"
    echo "    · 生成工程:      bash d.ops_develop/d.scaffold/a.msopgen/run.sh"
else
    echo -e "  ${YELLOW}⚠ NO-GO: 存在缺失项, 请先补齐后再开始算子开发。${RESET}"
    echo "  常见补齐方式:"
    echo "    · 缺 CANN: 在容器内安装 toolkit 或更换含 CANN 的镜像"
    echo "    · 缺 torch_npu: pip install torch_npu (版本与 torch/CANN 匹配)"
    echo "    · 缺编译链: apt install -y build-essential cmake"
fi
echo ""
INNER

echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${GREEN}检查完成。${RESET} 如需进入容器交互操作: docker exec -it $NAME bash"
echo ""
