#!/bin/bash
# ============================================================
# ① 服务器 CANN 环境检查 + 一键修复向导
# 第一步: 检查并汇总 主机/OS、驱动、CANN 安装目录与版本、激活状态、
#         Python/框架、编译链、Docker。
# 第二步: 根据缺失项给出「可选修复方案」, 用户只需选择方案编号(或 A 全部),
#         脚本自动执行, 无需用户手动逐条检查。
#   方案包括:
#     1) 安装 CANN (合一包 / toolkit+kernels / 仅 toolkit)
#     2) 生成/激活 CANN 环境脚本 activate_cann.sh (并可写入 ~/.bashrc)
#     3) 安装 torch + torch_npu (昇腾配套)
#     4) 安装 pybind11
#     5) 安装编译工具链 gcc/cmake
#     6) Docker 安装指引(打印命令)
# 非交互(无 TTY)时只打印建议步骤, 不执行。
# ============================================================
set -o pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

section() { echo; echo -e "  ${CYAN}===== $1 =====${RESET}"; }
have()    { command -v "$1" >/dev/null 2>&1; }
envval()  { eval "printf '%s' \"\${$1:-}\""; }
ok()      { echo -e "  ${GREEN}[ OK ]${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}[ WARN]${RESET} $1"; }
bad()     { echo -e "  ${RED}[缺失]${RESET} $1"; }
ask()     { local p="$1" d="$2"; printf "  %s [%s]: " "$p" "$d"; IFS= read -r REPLY || REPLY=""; [ -z "$REPLY" ] && REPLY="$d"; }
confirm() { local a; printf "  %s [y/N]: " "$1"; IFS= read -r a || a=""; case "$a" in y|Y|yes|YES) return 0;; *) return 1;; esac; }

# ---- 缺失标志 ----
MISS_TOOLKIT=0; MISS_SETENV=0; MISS_ACL=0; MISS_TORCH=0
MISS_TORCH_NPU=0; MISS_PYBIND11=0; MISS_BUILD=0; MISS_DOCKER=0

# 汇总收集器
G_DIR=""; G_VERSION=""; G_DRIVER=""; SETENV=""

echo ""
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  ① 服务器 CANN 环境检查${RESET}"
echo -e "  ${WHITE}════════════════════════════════════════════════════${RESET}"

# ============================================================
# 1. 主机与 OS
# ============================================================
section "1. 主机与 OS"
echo "  主机名   : $(hostname 2>/dev/null || echo unknown)"
echo "  内核     : $(uname -r 2>/dev/null)"
echo "  架构     : $(uname -m 2>/dev/null)"
( . /etc/os-release 2>/dev/null && echo "  发行版   : $PRETTY_NAME" ) || echo "  发行版   : (unknown)"
echo "  python3  : $(command -v python3 2>/dev/null || echo '(未找到)') $(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"

# ============================================================
# 2. 驱动(HDK)
# ============================================================
section "2. 驱动 (HDK / NPU 设备)"
if [ -f /etc/ascend_install.info ]; then
    G_DRIVER=$(grep -iE '^version|driver' /etc/ascend_install.info 2>/dev/null | head -1)
    ok "驱动安装信息存在: $(echo "$G_DRIVER" | head -1)"
else
    bad "未找到 /etc/ascend_install.info (可能未安装驱动)"
fi
have npu-smi && ok "npu-smi: $(npu-smi -v 2>/dev/null | head -1 || true)" || warn "未找到 npu-smi"
N_DEV=$(ls /dev/davinci* 2>/dev/null | wc -l | tr -d ' ')
[ "${N_DEV:-0}" -gt 0 ] 2>/dev/null && ok "检测到 ${N_DEV} 个 NPU 设备" || warn "未检测到 /dev/davinci* 设备"

# ============================================================
# 3. CANN 安装目录
# ============================================================
section "3. CANN 安装目录"
TOOLKIT_DIR=""
CANDIDATES=()
[ -n "$(envval ASCEND_HOME_PATH)" ] && CANDIDATES+=("$(envval ASCEND_HOME_PATH)")
[ -n "$(envval ASCEND_TOOLKIT_HOME)" ] && CANDIDATES+=("$(envval ASCEND_TOOLKIT_HOME)")
CANDIDATES+=("/usr/local/Ascend/ascend-toolkit/latest" "/usr/local/Ascend/ascend-toolkit" "/usr/local/Ascend")
FOUND_DIRS=()
for d in "${CANDIDATES[@]}"; do
    [ -n "$d" ] && [ -e "$d" ] && FOUND_DIRS+=("$d")
done
FOUND_DIRS=($(printf '%s\n' "${FOUND_DIRS[@]}" | awk '!seen[$0]++'))
if [ ${#FOUND_DIRS[@]} -eq 0 ]; then
    bad "未发现 CANN 安装目录"; MISS_TOOLKIT=1
else
    for d in "${FOUND_DIRS[@]}"; do
        [ -d "$d" ] && { echo -e "  ${GREEN}•${RESET} $d"; [ -n "$TOOLKIT_DIR" ] || TOOLKIT_DIR="$d"; }
    done
fi
G_DIR="$TOOLKIT_DIR"

# ============================================================
# 4. CANN 版本
# ============================================================
section "4. CANN 版本"
print_verfile() { if [ -f "$1" ]; then sed 's/^/        /' "$1" 2>/dev/null | head -12; fi; }
if [ -n "$TOOLKIT_DIR" ] && [ -d "$TOOLKIT_DIR" ]; then
    for vf in version.cfg version version.info; do print_verfile "$TOOLKIT_DIR/$vf"; done
    G_VERSION=$(grep -iE 'version' "$TOOLKIT_DIR/version.cfg" 2>/dev/null | head -1 | tr -d '[:space:]')
    [ -z "$G_VERSION" ] && G_VERSION=$(head -1 "$TOOLKIT_DIR/version" 2>/dev/null)
    [ -z "$G_VERSION" ] && G_VERSION="(见版本文件)"
else
    bad "无 toolkit 目录, 无法读取 CANN 版本"; MISS_TOOLKIT=1
fi
OPP_DIR="$(envval ASCEND_OPP_PATH)"
[ -z "$OPP_DIR" ] && [ -n "$TOOLKIT_DIR" ] && OPP_DIR="$TOOLKIT_DIR/opp"
if [ -n "$OPP_DIR" ] && [ -d "$OPP_DIR" ]; then
    echo -e "  ${GREEN}•${RESET} OPP(算子原型库): $OPP_DIR"; print_verfile "$OPP_DIR/version.info"
else
    warn "未找到 OPP(算子原型库) (算子开发需要 kernels)"
fi

# ============================================================
# 5. 环境变量激活状态
# ============================================================
section "5. 环境变量激活状态"
for v in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_AICPU_PATH ASCEND_RUNTIME_HOME ASCEND_DRIVER_PATH; do
    val="$(envval "$v")"
    [ -n "$val" ] && echo -e "  ${GREEN}[已设置]${RESET} $v = $val" || echo -e "  ${YELLOW}[未设置]${RESET} $v"
done
SETENV=""
for p in "$TOOLKIT_DIR/set_env.sh" "$(envval ASCEND_HOME_PATH)/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    [ -f "$p" ] && SETENV="$p" && break
done
if [ -n "$SETENV" ]; then ok "找到激活脚本: $SETENV"; else bad "未找到 set_env.sh"; MISS_SETENV=1; fi
LDHAS=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -i ascend | head -1)
[ -n "$LDHAS" ] && ok "LD_LIBRARY_PATH 已含 ascend 路径" || warn "LD_LIBRARY_PATH 未含 ascend 路径(当前 shell 未激活)"

# ============================================================
# 6. Python / 框架 / 编译链
# ============================================================
section "6. Python / 框架 / 编译链"
if python3 -c "import acl" 2>/dev/null; then ok "acl 可导入 (CANN Python 接口正常)"; else warn "acl 导入失败"; MISS_ACL=1; fi
if python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null; then ok "torch: $(python3 -c "import torch;print(torch.__version__)" 2>/dev/null)"; else bad "torch 未安装"; MISS_TORCH=1; fi
if python3 -c "import torch_npu; print('torch_npu', torch_npu.__version__)" 2>/dev/null; then ok "torch_npu: $(python3 -c "import torch_npu;print(torch_npu.__version__)" 2>/dev/null)"; else bad "torch_npu 未安装"; MISS_TORCH_NPU=1; fi
python3 -c "import pybind11" 2>/dev/null && ok "pybind11: $(python3 -c "import pybind11;print(pybind11.__version__)" 2>/dev/null)" || { warn "pybind11 未安装"; MISS_PYBIND11=1; }
if have cmake && have gcc; then ok "cmake/gcc 就绪"; else warn "缺少 cmake 或 gcc"; MISS_BUILD=1; fi

# ============================================================
# 7. Docker
# ============================================================
section "7. Docker"
have docker && ok "docker: $(docker --version 2>/dev/null)" || { warn "docker 未安装"; MISS_DOCKER=1; }

# ============================================================
# 汇总报告
# ============================================================
echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  【汇总报告】${RESET}"
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
printf '  %-16s: %s\n' "安装目录" "${G_DIR:-未发现}"
printf '  %-16s: %s\n' "CANN 版本" "${G_VERSION:-未知}"
printf '  %-16s: %s\n' "激活脚本" "${SETENV:-未找到}"
printf '  %-16s: %s\n' "驱动信息" "${G_DRIVER:-未检测到}"
printf '  %-16s: %s\n' "NPU 设备数" "${N_DEV:-0}"
echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"

# ============================================================
# 修复动作实现
# ============================================================
dl() { # $1=url $2=dest
    if have wget; then wget -c -O "$2" "$1"; elif have curl; then curl -L -C - -o "$2" "$1"; else echo -e "${RED}未找到 wget/curl。${RESET}" >&2; return 1; fi
}

fix_cann() {
    local method="${INSTALL_METHOD:-}" ver="${CANN_VERSION:-}" chip="${CHIP:-910b}" arch="${ARCH:-$(uname -m)}"
    echo -e "\n  ${CYAN}▶ 安装 CANN${RESET}"
    if [ -z "$method" ]; then
        echo "    安装方式: 1) toolkit+kernels(推荐)  2) 仅toolkit  3) 合一包"
        printf "    选择 [1]: "; IFS= read -r m || m=""; m="${m:-1}"
        case "$m" in 2) method=toolkit;; 3) method=combined;; *) method=all;; esac
    fi
    [ -z "$ver" ] && { ask "    CANN 版本" "8.1.RC1"; ver="$REPLY"; }
    case "$method" in
        all)      pkgs=("Ascend-cann-toolkit_${ver}_linux-${arch}.run" "Ascend-cann-kernels-${chip}_${ver}_linux-${arch}.run") ;;
        toolkit)  pkgs=("Ascend-cann-toolkit_${ver}_linux-${arch}.run") ;;
        combined) pkgs=("Ascend-cann-${ver}_linux-${arch}.run") ;;
    esac
    local base="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${ver}}"
    local pkgdir="${PKG_DIR:-$PWD/cann_pkgs}"; mkdir -p "$pkgdir"
    local install_dir="${INSTALL_DIR:-/usr/local/Ascend/ascend-toolkit}"
    [ "$method" = combined ] && install_dir="${INSTALL_DIR:-/usr/local/Ascend}"
    local p path
    for p in "${pkgs[@]}"; do
        path="$pkgdir/$p"
        [ -f "$path" ] || { echo -e "    ${CYAN}下载 $p ...${RESET}"; dl "${base}/${p}" "$path" || return 1; }
    done
    local sudo_cmd=""
    [ "$(id -u)" != "0" ] && have sudo && sudo_cmd="sudo"
    local qf=""; [ "${QUIET:-0}" = "1" ] && qf="--quiet"
    for p in "${pkgs[@]}"; do
        echo -e "    ${CYAN}安装 $p ...${RESET}"
        chmod +x "$pkgdir/$p" 2>/dev/null || true
        $sudo_cmd "$pkgdir/$p" --install --install-path="$install_dir" $qf || { echo -e "${RED}安装失败: $p${RESET}" >&2; return 1; }
    done
    # 重新定位 set_env.sh
    for s in "$install_dir/set_env.sh" "$install_dir/ascend-toolkit/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh; do
        [ -f "$s" ] && SETENV="$s" && break
    done
    MISS_TOOLKIT=0; MISS_SETENV=0
    echo -e "    ${GREEN}CANN 安装完成。${RESET}"
    return 0
}

fix_activation() {
    echo -e "\n  ${CYAN}▶ 生成/激活 CANN 环境脚本${RESET}"
    if [ -z "$SETENV" ]; then
        for s in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
            [ -f "$s" ] && SETENV="$s" && break
        done
    fi
    if [ -z "$SETENV" ]; then
        echo -e "    ${RED}仍未找到 set_env.sh, 请先选择方案「安装 CANN」。${RESET}" >&2; return 1
    fi
    local act="$PWD/activate_cann.sh"
    cat > "$act" <<ACT_EOF
#!/bin/bash
# ============================================================
# CANN 环境激活脚本 (由 itool 生成)
# 用法: source $act    或    . $act
# 作用: 加载 CANN 的 PATH / LD_LIBRARY_PATH / PYTHONPATH 等
# ============================================================
source "$SETENV"
echo "[activate_cann] 已激活: $SETENV"
ACT_EOF
    chmod +x "$act"
    echo -e "    ${GREEN}已生成: $act${RESET}"
    echo -e "    立即激活并验证 acl ..."
    ( source "$SETENV" 2>/dev/null; python3 -c "import acl; print('      acl OK, soc:', acl.get_soc_name())" 2>/dev/null ) \
        && { echo -e "    ${GREEN}acl 验证通过。${RESET}"; MISS_ACL=0; } \
        || echo -e "    ${YELLOW}acl 验证失败(可能缺 python 包)。${RESET}"
    if confirm "    是否写入 ~/.bashrc 永久激活?"; then
        if ! grep -qF "source $SETENV" "$HOME/.bashrc" 2>/dev/null; then
            echo "source $SETENV" >> "$HOME/.bashrc"
            echo -e "    ${GREEN}已写入 ~/.bashrc${RESET}"
        else
            echo -e "    ${YELLOW}~/.bashrc 已存在该配置, 跳过。${RESET}"
        fi
    fi
    return 0
}

fix_torch() {
    echo -e "\n  ${CYAN}▶ 安装 torch + torch_npu${RESET}"
    local tv="${TORCH_VER:-}"
    [ -z "$tv" ] && { ask "    torch 版本(2.1.0 / 2.6.0)" "2.6.0"; tv="$REPLY"; }
    local py=$(python3 -c 'import sys;print("%d%d"%(sys.version_info[0],sys.version_info[1]))' 2>/dev/null)
    local arch=$(uname -m)
    local torch_npu_url=""
    if [ "$tv" = "2.1.0" ]; then
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu/ || return 1
        if [ "$arch" = x86_64 ]; then torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.1.0/torch_npu-2.1.0.post16-cp${py}-cp${py}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        else torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.1.0/torch_npu-2.1.0.post16-cp${py}-cp${py}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"; fi
    else
        pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu/ || return 1
        if [ "$arch" = x86_64 ]; then torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp${py}-cp${py}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
        else torch_npu_url="https://gitee.com/ascend/pytorch/releases/download/v7.1.0.2-pytorch2.6.0/torch_npu-2.6.0.post2-cp${py}-cp${py}-manylinux_2_28_aarch64.whl"; fi
    fi
    echo -e "    ${CYAN}pip install torch_npu ...${RESET}"
    pip install "$torch_npu_url" || return 1
    MISS_TORCH=0; MISS_TORCH_NPU=0
    echo -e "    ${GREEN}torch / torch_npu 安装完成。${RESET}"
    return 0
}

fix_pybind11() {
    echo -e "\n  ${CYAN}▶ 安装 pybind11${RESET}"
    pip install pybind11 || { echo -e "${YELLOW}pip 失败, 尝试 apt ...${RESET}"; $sudo apt-get install -y pybind11-dev 2>/dev/null || true; }
    MISS_PYBIND11=0
    echo -e "    ${GREEN}pybind11 完成。${RESET}"
}

fix_build() {
    echo -e "\n  ${CYAN}▶ 安装编译工具链 gcc/cmake${RESET}"
    if [ "$(id -u)" = "0" ]; then apt-get update -y && apt-get install -y build-essential cmake 2>/dev/null; else have sudo && sudo apt-get update -y && sudo apt-get install -y build-essential cmake 2>/dev/null; fi
    have gcc || pip install cmake 2>/dev/null || true
    MISS_BUILD=0
    echo -e "    ${GREEN}编译工具链处理完成。${RESET}"
}

fix_docker() {
    echo -e "\n  ${CYAN}▶ Docker 安装指引(需 root, 请手动执行)${RESET}"
    cat <<DOCKER_GUIDE
    # Ubuntu/Debian:
    sudo apt-get update && sudo apt-get install -y docker.io
    # CentOS/RHEL:
    sudo yum install -y docker && sudo systemctl start docker
    # 验证:
    docker --version && docker run --rm hello-world
DOCKER_GUIDE
}

# ============================================================
# 修复向导
# ============================================================
NEEDED=$(( MISS_TOOLKIT + MISS_SETENV + MISS_ACL + MISS_TORCH + MISS_TORCH_NPU + MISS_PYBIND11 + MISS_BUILD + MISS_DOCKER ))

if [ "$NEEDED" -eq 0 ]; then
    echo -e "  ${GREEN}✅ 环境完整, 无需修复, 可以开始算子开发。${RESET}"
    echo -e "    下一步: ③ 拉取镜像 → bash d.ops_develop/b.env_setup/a.pull_image/run.sh"
    exit 0
fi

echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${WHITE}  【可选修复方案】(选择编号执行, 无需手动检查)${RESET}"
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
[ "$MISS_TOOLKIT" = "1" ] && echo -e "    ${RED}[1]${RESET} 安装 CANN (当前缺失)"
[ "$MISS_SETENV" = "1" -o -n "$SETENV" ] && echo -e "    ${GREEN}[2]${RESET} 生成/激活 CANN 环境脚本 activate_cann.sh"
[ "$MISS_TORCH" = "1" -o "$MISS_TORCH_NPU" = "1" ] && echo -e "    ${RED}[3]${RESET} 安装 torch + torch_npu"
[ "$MISS_PYBIND11" = "1" ] && echo -e "    ${RED}[4]${RESET} 安装 pybind11"
[ "$MISS_BUILD" = "1" ] && echo -e "    ${RED}[5]${RESET} 安装编译工具链 gcc/cmake"
[ "$MISS_DOCKER" = "1" ] && echo -e "    ${YELLOW}[6]${RESET} Docker 安装指引"
echo -e "    ${WHITE}[A]${RESET} 一键修复以上全部缺失项"
echo -e "    ${WHITE}[0]${RESET} 跳过(仅查看)"
echo ""

if [ ! -t 0 ]; then
    echo -e "  ${YELLOW}非交互终端, 已跳过自动修复。请手动执行上述编号对应的脚本/命令。${RESET}"
    exit 0
fi

printf "  请选择方案 [A]: "
IFS= read -r choice || choice=""
choice="${choice:-A}"
choice=$(printf '%s' "$choice" | tr '[:lower:]' '[:upper:]')

run_plan() {
    # 按顺序执行: CANN -> 激活 -> torch -> pybind11 -> build -> docker(仅打印)
    [ "$MISS_TOOLKIT" = "1" ] && fix_cann
    [ -n "$SETENV" ] && fix_activation
    [ "$MISS_TORCH" = "1" -o "$MISS_TORCH_NPU" = "1" ] && fix_torch
    [ "$MISS_PYBIND11" = "1" ] && fix_pybind11
    [ "$MISS_BUILD" = "1" ] && fix_build
    [ "$MISS_DOCKER" = "1" ] && fix_docker
}

case "$choice" in
    A) run_plan ;;
    0) echo -e "  ${YELLOW}已跳过修复。${RESET}" ;;
    1) fix_cann ;;
    2) fix_activation ;;
    3) fix_torch ;;
    4) fix_pybind11 ;;
    5) fix_build ;;
    6) fix_docker ;;
    *) echo -e "  ${YELLOW}无效选择, 已跳过。${RESET}" ;;
esac

echo ""
echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
echo -e "  ${GREEN}检查与修复流程结束。${RESET}"
[ -n "$SETENV" ] && echo -e "  激活环境: ${WHITE}source $SETENV${RESET}  或  source \$PWD/activate_cann.sh"
echo -e "  下一步: ③ 拉取镜像 → bash d.ops_develop/b.env_setup/a.pull_image/run.sh"
echo ""
