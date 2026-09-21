#!/bin/bash
# ============================================================
# ⑧-B 离线 Skill 仓库（选装）
# 从本地 skills.tar.gz 解压出全部 skill 到仓库目录，交互式按模块
# 浏览、勾选，把选中的 skill 复制进 opencode 的 ~/.config/opencode/skills/
#
# 说明:
#   - 全离线，不依赖网络（解决客户机连不上 gitcode 的问题）
#   - skill 已带模块前缀(如 cannbot-xxx / mindcluster-xxx)，name 与目录名一致
#   - 已装 skill 可查看、可卸载
#
# 用法:
#   bash b.skill_store/run.sh            # 交互式选装
#   bash b.skill_store/run.sh list       # 列出仓库内全部 skill（按模块）
#   bash b.skill_store/run.sh status     # 查看已装 skill
#   bash b.skill_store/run.sh install <skill名>   # 直接安装指定 skill
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
TARBALL="$SCRIPT_DIR/skills.tar.gz"
REPO_DIR="$SCRIPT_DIR/repo"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

CONFIG_SKILLS="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills"
mkdir -p "$CONFIG_SKILLS"

# ---------- 工具函数 ----------
# 解压仓库（幂等）
ensure_repo() {
    if [ ! -d "$REPO_DIR" ] || [ -z "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]; then
        if [ ! -f "$TARBALL" ]; then
            echo -e "${RED}[ERROR] 找不到 skills.tar.gz: $TARBALL${RESET}" >&2
            exit 1
        fi
        echo "==> 解压离线 skill 仓库..."
        mkdir -p "$REPO_DIR"
        tar -xzf "$TARBALL" -C "$REPO_DIR"
        echo "==> 解压完成"
    else
        echo "==> skill 仓库已就绪: $REPO_DIR"
    fi
}

# 列出仓库内所有 skill（排序）；结果放 ALL_SKILLS 数组
list_repo() {
    ALL_SKILLS=()
    while IFS= read -r -d '' d; do
        [ -f "$d/SKILL.md" ] || continue
        ALL_SKILLS+=("$(basename "$d")")
    done < <(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
}

# 按模块分组打印（只读）
print_modules() {
    local -a names=("${ALL_SKILLS[@]}")
    if [ ${#names[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}（空）${RESET}"
        return
    fi
    # 模块 = skill 名的前缀（去掉最后一个 -xxx 的尾段不可靠，这里取前两段连字符之前的模块词根）
    # 简化：直接列出全部，按前缀字母排序
    printf '  %s\n' "${names[@]}"
}

# 根据 skill 名取其模块前缀（去掉末尾具体名）
mod_of() {
    local n="$1"
    echo "$n" | sed -E 's/-[^-]+$//'
}

# 列出某个模块下的 skill；结果放 MOD_SKILLS
list_module() {
    local mod="$1"
    MOD_SKILLS=()
    local s
    for s in "${ALL_SKILLS[@]}"; do
        if [ "$(mod_of "$s")" = "$mod" ]; then
            MOD_SKILLS+=("$s")
        fi
    done
}

# 打印 skill 的 description（一句话）
desc_of() {
    local name="$1" d
    d="$REPO_DIR/$name"
    if [ -f "$d/SKILL.md" ]; then
        python3 -c 'import re,sys; t=open(sys.argv[1],encoding="utf-8",errors="replace").read(); m=re.search(r"description:\s*(.+)", t, re.S); print(m.group(1).strip()[:120] if m else "(无描述)")' "$d/SKILL.md" 2>/dev/null || echo "(读取失败)"
    else
        echo "(不存在)"
    fi
}

# 是否已装
is_installed() {
    [ -d "$CONFIG_SKILLS/$1" ]
}

# 安装单个 skill（复制目录）
install_one() {
    local name="$1"
    if [ ! -d "$REPO_DIR/$name" ]; then
        echo -e "${RED}[ERROR] 仓库中不存在 skill: $name${RESET}" >&2
        return 1
    fi
    if is_installed "$name"; then
        echo -e "  ${YELLOW}已安装，跳过: $name${RESET}"
        return 0
    fi
    cp -r "$REPO_DIR/$name" "$CONFIG_SKILLS/$name"
    echo -e "  ${GREEN}✔ 已安装: $name${RESET}"
}

# 卸载单个 skill
uninstall_one() {
    local name="$1"
    if is_installed "$name"; then
        rm -rf "$CONFIG_SKILLS/$name"
        echo -e "  ${GREEN}✔ 已卸载: $name${RESET}"
    else
        echo -e "  ${YELLOW}未安装: $name${RESET}"
    fi
}

# 交互式选装主流程
interactive_install() {
    list_repo
    if [ ${#ALL_SKILLS[@]} -eq 0 ]; then
        echo -e "${RED}仓库为空。${RESET}" >&2
        return 1
    fi
    # 收集所有模块名
    MODS=()
    local s m
    for s in "${ALL_SKILLS[@]}"; do
        m=$(mod_of "$s")
        if [[ ! " ${MODS[@]} " =~ " $m " ]]; then
            MODS+=("$m")
        fi
    done
    # 排序模块
    MODS=($(printf '%s\n' "${MODS[@]}" | sort))

    while true; do
        echo ""
        echo -e "  ${WHITE}===== 离线 Skill 选装 =====${RESET}"
        echo ""
        echo -e "  ${CYAN}选择模块：${RESET}"
        local i
        for i in "${!MODS[@]}"; do
            local cnt=0 s2
            for s2 in "${ALL_SKILLS[@]}"; do
                [ "$(mod_of "$s2")" = "${MODS[$i]}" ] && cnt=$((cnt+1))
            done
            printf '    %3d) %-20s (%d 个)\n' "$((i+1))" "${MODS[$i]}" "$cnt"
        done
        echo -e "    ${GREEN}  a${RESET}) 全部安装"
        echo -e "    ${GREEN}  q${RESET}) 退出"
        echo ""
        printf "  请选择模块编号 [q]: "
        IFS= read -r ans || ans="q"
        ans="${ans:-q}"
        case "$ans" in
            q|Q) return 0 ;;
            a|A)
                for s2 in "${ALL_SKILLS[@]}"; do install_one "$s2"; done
                ;;
            *)
                if [ "$ans" -ge 1 ] 2>/dev/null && [ "$ans" -le "${#MODS[@]}" ] 2>/dev/null; then
                    select_in_module "${MODS[$((ans-1))]}"
                else
                    echo -e "  ${RED}无效选择。${RESET}"
                fi
                ;;
        esac
    done
}

# 在模块内勾选 skill
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
        echo "                $(desc_of "$s")"
    done
    echo -e "    ${GREEN}  a${RESET}) 全选本模块"
    echo -e "    ${GREEN}  b${RESET}) 返回"
    echo ""
    printf "  输入编号(可逗号分隔多选, 如 1,3,5): "
    IFS= read -r ans || ans="b"
    case "$ans" in
        b|B) return 0 ;;
        a|A) for s in "${MOD_SKILLS[@]}"; do install_one "$s"; done ;;
        *)
            local part idx nums=()
            for part in $(echo "$ans" | tr ',' ' '); do
                [ -n "$part" ] && nums+=("$part")
            done
            for idx in "${nums[@]}"; do
                if [ "$idx" -ge 1 ] 2>/dev/null && [ "$idx" -le "${#MOD_SKILLS[@]}" ] 2>/dev/null; then
                    install_one "${MOD_SKILLS[$((idx-1))]}"
                fi
            done
            ;;
    esac
}

# 查看已装状态
show_status() {
    echo ""
    echo -e "  ${WHITE}===== 已安装 skill ====="
    echo ""
    if [ -d "$CONFIG_SKILLS" ]; then
        local n=0
        local s
        while IFS= read -r -d '' d; do
            n=$((n+1))
            printf '    %3d) %s\n' "$n" "$(basename "$d")"
        done < <(find "$CONFIG_SKILLS" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null | sort -z)
        echo ""
        echo -e "  共 ${n} 个已安装"
    else
        echo "  (无)"
    fi
}

# ============================================================
CMD="${1:-}"

ensure_repo
list_repo

case "$CMD" in
    list)
        echo ""
        echo -e "  ${WHITE}===== 离线仓库 skill 列表（${#ALL_SKILLS[@]} 个）====="
        echo ""
        printf '  %s\n' "${ALL_SKILLS[@]}"
        ;;
    status)
        show_status
        ;;
    install)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}用法: $0 install <skill名>${RESET}" >&2
            exit 1
        fi
        install_one "$2"
        ;;
    uninstall)
        if [ -z "${2:-}" ]; then
            echo -e "${RED}用法: $0 uninstall <skill名>${RESET}" >&2
            exit 1
        fi
        uninstall_one "${2:-}"
        ;;
    "")
        echo ""
        echo -e "  ${WHITE}===== 离线 Skill 仓库 =====${RESET}"
        echo "  仓库 skill 总数 : ${#ALL_SKILLS[@]}"
        echo "  解压目录        : $REPO_DIR"
        echo "  目标目录        : $CONFIG_SKILLS"
        echo ""
        interactive_install
        ;;
    *)
        echo "用法: $0 [list|status|install <名>|uninstall <名>]（无参数进入交互式选装）" >&2
        exit 1
        ;;
esac