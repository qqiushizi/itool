#!/bin/bash
# ============================================================
# 大模型配置公共函数
# 供 d.ops_develop 下需要 LLM 功能的 run.sh source 使用
# ============================================================

# 解析仓库根目录和工作区
resolve_ops_root() {
    local SCRIPT_DIR base_dir="${1:-${ITOOL_SCRIPT_DIR:-$PWD}}"
    SCRIPT_DIR=$(cd "$(dirname "$base_dir")" 2>/dev/null && pwd || echo "$PWD")
    REPO_ROOT="${ITOOL_REPO_ROOT:-}"
    if [ -z "$REPO_ROOT" ] && [ -f "$PWD/itool.sh" ]; then
        REPO_ROOT="$PWD"
    fi
    if [ -z "$REPO_ROOT" ] && command -v git >/dev/null 2>&1; then
        REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
    fi
    if [ -z "$REPO_ROOT" ]; then
        local d="$SCRIPT_DIR"
        while [ "$d" != "/" ]; do
            if [ -f "$d/itool.sh" ]; then REPO_ROOT="$d"; break; fi
            d=$(dirname "$d")
        done
    fi
    [ -z "$REPO_ROOT" ] && REPO_ROOT="$SCRIPT_DIR"
    OPS_WORKSPACE="${OPS_WORKSPACE:-$REPO_ROOT/d.ops_develop/workspace}"
    LLM_PROFILE_DIR="$OPS_WORKSPACE/llm_configs"
    LLM_CURRENT_FILE="$LLM_PROFILE_DIR/current"
    export REPO_ROOT OPS_WORKSPACE LLM_PROFILE_DIR LLM_CURRENT_FILE
    mkdir -p "$LLM_PROFILE_DIR"
}

# 列出已有配置；结果放 PROFILE_FILES / PROFILE_NAMES
list_llm_profiles() {
    PROFILE_FILES=()
    PROFILE_NAMES=()
    local f name
    for f in "$LLM_PROFILE_DIR"/*.json; do
        [ -e "$f" ] || continue
        name=$(basename "$f" .json)
        PROFILE_FILES+=("$f")
        PROFILE_NAMES+=("$name")
    done
}

# 选择一个配置；结果放 SELECTED_PROFILE / SELECTED_PROFILE_FILE
choose_llm_profile() {
    list_llm_profiles
    SELECTED_PROFILE=""
    SELECTED_PROFILE_FILE=""
    if [ ${#PROFILE_NAMES[@]} -eq 0 ]; then
        [ -n "${ITOOL_LLM_PROFILE:-}" ] || return 1
    fi
    if [ -n "${ITOOL_LLM_PROFILE:-}" ] && [ -f "$LLM_PROFILE_DIR/${ITOOL_LLM_PROFILE}.json" ]; then
        SELECTED_PROFILE="$ITOOL_LLM_PROFILE"
        SELECTED_PROFILE_FILE="$LLM_PROFILE_DIR/$SELECTED_PROFILE.json"
        return 0
    fi
    local current_default=1
    if [ -f "$LLM_CURRENT_FILE" ]; then
        local cur
        cur=$(cat "$LLM_CURRENT_FILE" 2>/dev/null || true)
        for ((i=0; i<${#PROFILE_NAMES[@]}; i++)); do
            if [ "${PROFILE_NAMES[$i]}" = "$cur" ]; then current_default=$((i+1)); break; fi
        done
    fi
    echo ""
    echo -e "  请选择大模型配置:"
    for ((i=0; i<${#PROFILE_NAMES[@]}; i++)); do
        printf '    %3d) %s\n' "$((i+1))" "${PROFILE_NAMES[$i]}"
    done
    local n
    while true; do
        printf "  请选择配置编号 [%s]: " "$current_default"
        IFS= read -r n || n=""
        [ -z "$n" ] && n="$current_default"
        if [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le "${#PROFILE_NAMES[@]}" ] 2>/dev/null; then
            SELECTED_PROFILE="${PROFILE_NAMES[$((n-1))]}"
            SELECTED_PROFILE_FILE="${PROFILE_FILES[$((n-1))]}"
            return 0
        fi
        echo "  编号无效。"
    done
}

# 把配置读入 PROVIDER/API_BASE/API_KEY/MODEL/TIMEOUT
load_llm_profile() {
    local p f
    p="${1:-}"
    f="$LLM_PROFILE_DIR/$p.json"
    [ -f "$f" ] || { echo "配置文件不存在: $f" >&2; return 1; }
    while IFS='=' read -r k v; do
        case "$k" in
            provider) PROVIDER="$v" ;;
            api_base) API_BASE="$v" ;;
            api_key)   API_KEY="$v" ;;
            model)     MODEL="$v" ;;
            timeout)   TIMEOUT="$v" ;;
        esac
    done < <(python3 - "$f" <<'PY'
import json, sys
path=sys.argv[1]
with open(path, encoding='utf-8') as fp:
    d=json.load(fp)
for k in ['provider','api_base','api_key','model','timeout']:
    v=d.get(k,'')
    if isinstance(v,(int,float)):
        v=str(v)
    print('%s=%s' % (k, v))
PY
)
    TIMEOUT="${TIMEOUT:-120}"
}
