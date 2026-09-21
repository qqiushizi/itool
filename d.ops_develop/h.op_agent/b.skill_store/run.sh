#!/bin/bash
# ============================================================
# ⑧-B 离线 Skill 仓库（选装）
# 从本地 skills.tar.gz 解压出全部 skill（按模块目录分类），交互式
# 按模块浏览、勾选，把选中的 skill 按【原名】复制进 opencode 的
# skills 目录（同名直接覆盖，可更新 tar 包自带的旧 skill）。
#
# 目录约定:
#   仓库(repo/)        : <模块>/<skill原名>    （给人看，分类清晰）
#   opencode skills/   : <skill原名>           （原名平铺，不拼前缀）
#
# 说明:
#   - 全离线，不依赖网络
#   - 安装目标 = opencode 绿色包内的 config/opencode/skills/
#     （便携版 opencode 通过 start.sh 把 XDG_CONFIG_HOME 指到 config/）
#
# 用法:
#   bash b.skill_store/run.sh            # 交互式选装
#   bash b.skill_store/run.sh list       # 列出仓库全部 skill（按模块）
#   bash b.skill_store/run.sh status     # 查看已装 skill
#   bash b.skill_store/run.sh install <skill原名>   # 直接安装指定 skill
#   bash b.skill_store/run.sh uninstall <skill原名> # 卸载
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
A_INSTALL_DIR="$SCRIPT_DIR/../a.install_opencode"
TARBALL="$SCRIPT_DIR/skills.tar.gz"
REPO_DIR="$SCRIPT_DIR/repo"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

# ---------- 定位 opencode 的 skills 目录 ----------
# 便携版 opencode 绿色包解压在 a.install_opencode/opencode/，
# 其 start.sh 里 export XDG_CONFIG_HOME=<opencode>/config，
# 因此 opencode 实际读取 config/opencode/skills/
resolve_opencode_skills() {
    OPCODE_ROOT=""
    if [ -d "$A_INSTALL_DIR/opencode" ]; then
        OPCODE_ROOT="$A_INSTALL_DIR/opencode"
    else
        echo -e "${YELLOW}[提示] 尚未解压 opencode 绿色包。${RESET}"
        echo "  请先运行: bash $A_INSTALL_DIR/run.sh"
        return 1
    fi
    CONFIG_SKILLS="$OPCODE_ROOT/config/opencode/skills"
    mkdir -p "$CONFIG_SKILLS"
    return 0
}

# ---------- 仓库 ----------
ensure_repo() {
    if [ ! -d "$REPO_DIR" ] || [ -z "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]; then
        if [ ! -f "$TARBALL" ]; then
            echo -e "${RED}[ERROR] 找不到 skills.tar.gz: $TARBALL${RESET}" >&2
            exit 1
        fi
        echo "==> 解压离线 skill 仓库..."
        mkdir -p "$REPO_DIR"
        tar -xzf "$TARBALL" -C "$REPO_DIR"
    fi
    # 列出模块（仓库下的一级目录，排除 entries.json）
    MODULES=()
    while IFS= read -r -d '' d; do
        MODULES+=("$(basename "$d")")
    done < <(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
}

# 列出某模块下的 skill 名（原名）；结果放 MOD_SKILLS
list_module() {
    local mod="$1"
    MOD_SKILLS=()
    local d
    while IFS= read -r -d '' d; do
        [ -f "$d/SKILL.md" ] && MOD_SKILLS+=("$(basename "$d")")
    done < <(find "$REPO_DIR/$mod" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
}

# 打印 skill 的 description
desc_of() {
    local mod="$1" name="$2" d
    d="$REPO_DIR/$mod/$name/SKILL.md"
    if [ -f "$d" ]; then
        python3 -c 'import re,sys; t=open(sys.argv[1],encoding="utf-8",errors="replace").read(); m=re.search(r"description:\s*(.+)", t, re.S); print(m.group(1).strip()[:100] if m else "(无描述)")' "$d" 2>/dev/null || echo "(读取失败)"
    else
        echo "(不存在)"
    fi
}

is_installed() {
    [ -d "$CONFIG_SKILLS/$1" ]
}

# 安装单个 skill（按原名，覆盖）
install_one() {
    local mod="$1" name="$2"
    local src="$REPO_DIR/$mod/$name"
    if [ ! -d "$src" ]; then
        echo -e "${RED}[ERROR] 仓库中不存在: $mod/$name${RESET}" >&2
        return 1
    fi
    if is_installed "$name"; then
        rm -rf "$CONFIG_SKILLS/$name"
        echo -e "  ${YELLOW}覆盖更新: $name${RESET}"
    fi
    cp -r "$src" "$CONFIG_SKILLS/$name"
    echo -e "  ${GREEN}✔ 已安装: $name${RESET}"
}

uninstall_one() {
    local name="$1"
    if is_installed "$name"; then
        rm -rf "$CONFIG_SKILLS/$name"
        echo -e "  ${GREEN}✔ 已卸载: $name${RESET}"
    else
        echo -e "  ${YELLOW}未安装: $name${RESET}"
    fi
}

# ---------- 交互式 ----------
interactive_install() {
    while true; do
        echo ""
        echo -e "  ${WHITE}===== 离线 Skill 选装 =====${RESET}"
        echo ""
        echo -e "  ${CYAN}选择模块：${RESET}"
        local i
        for i in "${!MODULES[@]}"; do
            local cnt=0
            local d2
            for d2 in "$REPO_DIR/${MODULES[$i]}"/*/; do
                [ -d "$d2" ] && [ -f "$d2/SKILL.md" ] && cnt=$((cnt+1))
            done
            printf '    %3d) %-18s (%d 个)\n' "$((i+1))" "${MODULES[$i]}" "$cnt"
        done
        echo -e "    ${GREEN}  q${RESET}) 退出"
        echo ""
        printf "  请选择模块编号 [q]: "
        IFS= read -r ans || ans="q"
        ans="${ans:-q}"
        case "$ans" in
            q|Q) return 0 ;;
            *)
                if [ "$ans" -ge 1 ] 2>/dev/null && [ "$ans" -le "${#MODULES[@]}" ] 2>/dev/null; then
                    select_in_module "${MODULES[$((ans-1))]}"
                else
                    echo -e "  ${RED}无效选择。${RESET}"
                fi
                ;;
        esac
    done
}

select_in_module() {
    local mod="$1"
    list_module "$mod"
    echo ""
    echo -e "  ${WHITE}模块 $mod 下的 skill：${RESET}"
    local i s
    for i in "${!MOD_SKILLS[@]}"; do
        s="${MOD_SKILLS[$i]}"
        local mark=" "
        is_installed "$s" && mark="*"
        printf '    [%s] %3d) %s\n' "$mark" "$((i+1))" "$s"
        echo "                $(desc_of "$mod" "$s")"
    done
    echo -e "    ${GREEN}  a${RESET}) 全选本模块"
    echo -e "    ${GREEN}  b${RESET}) 返回"
    echo ""
    printf "  输入编号(可逗号分隔多选, 如 1,3,5): "
    IFS= read -r ans || ans="b"
    case "$ans" in
        b|B) return 0 ;;
        a|A) for s in "${MOD_SKILLS[@]}"; do install_one "$mod" "$s"; done ;;
        *)
            local part
            for part in $(echo "$ans" | tr ',' ' '); do
                [ -n "$part" ] || continue
                if [ "$part" -ge 1 ] 2>/dev/null && [ "$part" -le "${#MOD_SKILLS[@]}" ] 2>/dev/null; then
                    install_one "$mod" "${MOD_SKILLS[$((part-1))]}"
                fi
            done
            ;;
    esac
}

show_status() {
    echo ""
    echo -e "  ${WHITE}===== 已安装 skill ====="
    echo ""
    if [ -d "$CONFIG_SKILLS" ]; then
        local n=0
        while IFS= read -r -d '' d; do
            n=$((n+1))
            printf '    %3d) %s\n' "$n" "$(basename "$d")"
        done < <(find "$CONFIG_SKILLS" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
        echo ""
        echo -e "  共 ${n} 个"
    else
        echo "  (无)"
    fi
}

# ---------- 主流程 ----------
CMD="${1:-}"

if ! resolve_opencode_skills; then
    exit 1
fi
ensure_repo

case "$CMD" in
    list)
        echo ""
        echo -e "  ${WHITE}===== 离线仓库 skill（按模块）====="
        echo ""
        for m in "${MODULES[@]}"; do
            list_module "$m"
            echo -e "  ${CYAN}[$m] (${#MOD_SKILLS[@]})${RESET}"
            printf '      %s\n' "${MOD_SKILLS[@]}"
        done
        ;;
    status)
        show_status
        ;;
    install)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}用法: $0 install <skill原名>${RESET}" >&2
            exit 1
        fi
        local found=0
        for m2 in "${MODULES[@]}"; do
            if [ -d "$REPO_DIR/$m2/${2:-}" ]; then
                install_one "$m2" "${2:-}"
                found=1
                break
            fi
        done
        [ "$found" = "1" ] || echo -e "${RED}[ERROR] 仓库中找不到 skill: ${2:-}${RESET}" >&2
        ;;
    uninstall)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}用法: $0 uninstall <skill原名>${RESET}" >&2
            exit 1
        fi
        uninstall_one "${2:-}"
        ;;
    "")
        echo ""
        echo -e "  ${WHITE}===== 离线 Skill 仓库 =====${RESET}"
        echo "  模块数     : ${#MODULES[@]}"
        echo "  安装目标   : $CONFIG_SKILLS"
        echo ""
        interactive_install
        ;;
    *)
        echo "用法: $0 [list|status|install <名>|uninstall <名>]（无参数进入交互式选装）" >&2
        exit 1
        ;;
esac