#!/bin/bash
# ============================================================
# ① 大模型配置管理
#
# 功能:
#   创建 / 修改 / 删除 / 列出 / 测试大模型接入配置
#   配置保存到 workspace/llm_configs/<name>.json
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
source "$SCRIPT_DIR/../llm_profile.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

ask() {
    local p="$1" d="$2"
    printf "  %s [%s]: " "$p" "$d"
    IFS= read -r REPLY || REPLY=""
    [ -n "$REPLY" ] || REPLY="$d"
}
read_secret() {
    local p="$1" d="$2"
    printf "  %s [%s]: " "$p" "$d"
    stty -echo 2>/dev/null || true
    IFS= read -r REPLY || REPLY=""
    stty echo 2>/dev/null || true
    echo ""
    [ -n "$REPLY" ] || REPLY="$d"
}

resolve_ops_root "$SCRIPT_DIR"

print_config_list() {
    list_llm_profiles
    if [ ${#PROFILE_NAMES[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}当前没有大模型配置。${RESET}"
        return 0
    fi
    echo -e "  ${CYAN}已有配置:${RESET}"
    local i cur=""
    [ -f "$LLM_CURRENT_FILE" ] && cur=$(cat "$LLM_CURRENT_FILE" 2>/dev/null || true)
    for ((i=0; i<${#PROFILE_NAMES[@]}; i++)); do
        if [ "${PROFILE_NAMES[$i]}" = "$cur" ]; then
            printf '    %3d) %s  ★ 当前使用\n' "$((i+1))" "${PROFILE_NAMES[$i]}"
        else
            printf '    %3d) %s\n' "$((i+1))" "${PROFILE_NAMES[$i]}"
        fi
    done
}

save_profile_json() {
    local file="$1" provider="$2" api_base="$3" api_key="$4" model="$5" timeout="$6"
    PROVIDER="$provider" API_BASE="$api_base" API_KEY="$api_key" MODEL="$model" TIMEOUT="$timeout" \
    FILE="$file" python3 - <<'PY'
import os, json

def clean_text(v):
    # Bash -> Python 环境变量可能携带 surrogateescape 产生的非法字符，
    # 直接 json.dump(ensure_ascii=False) 会报 UnicodeEncodeError。
    return v.encode('utf-8', 'replace').decode('utf-8')

obj = {
  'provider': clean_text(os.environ['PROVIDER']),
  'api_base': clean_text(os.environ['API_BASE']),
  'api_key': clean_text(os.environ['API_KEY']),
  'model': clean_text(os.environ['MODEL']),
  'timeout': int(os.environ['TIMEOUT'] or '120'),
}
with open(clean_text(os.environ['FILE']), 'w', encoding='utf-8') as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
print('已保存: %s' % clean_text(os.environ['FILE']))
PY
}

create_or_edit_action() {
    local mode="$1" profile_name="" file=""
    local PROVIDER API_BASE API_KEY MODEL TIMEOUT
    PROVIDER="local"; API_BASE=""; API_KEY=""; MODEL=""; TIMEOUT="120"
    if [ "$mode" = "edit" ]; then
        if ! choose_llm_profile; then echo -e "  ${RED}没有可修改的配置。${RESET}" >&2; return 1; fi
        profile_name="$SELECTED_PROFILE"
        file="$SELECTED_PROFILE_FILE"
        load_llm_profile "$profile_name"
    else
        while true; do
            ask "配置名称(用于选择，如 deepseek / local_vllm)" ""
            profile_name="$REPLY"
            [ -n "$profile_name" ] && break
        done
        file="$LLM_PROFILE_DIR/$profile_name.json"
        if [ -f "$file" ]; then
            echo -e "  ${YELLOW}配置已存在，将覆盖。${RESET}"
        fi
    fi
    echo ""
    echo -e "  ${CYAN}请选择接入方式${RESET}"
    echo -e "    ${GREEN}A${RESET}) 本地大模型 (vLLM-ascend)"
    echo -e "    ${GREEN}B${RESET}) 外部 API"
    while true; do
        ask "请选择" "A"
        case "$REPLY" in
            a|A|1) PROVIDER="local"; break ;;
            b|B|2) PROVIDER="external"; break ;;
        esac
    done
    if [ "$PROVIDER" = "external" ]; then
        ask "API Base" "${API_BASE:-https://api.deepseek.com/v1}"; API_BASE="$REPLY"
        ask "Model" "${MODEL:-deepseek-chat}"; MODEL="$REPLY"
        read_secret "API Key" "$API_KEY"; API_KEY="$REPLY"
    else
        ask "API Base" "${API_BASE:-http://127.0.0.1:8000/v1}"; API_BASE="$REPLY"
        ask "Model" "${MODEL:-Qwen/Qwen2.5-7B-Instruct}"; MODEL="$REPLY"
        read_secret "API Key(本地可留空)" "$API_KEY"; API_KEY="$REPLY"
    fi
    API_BASE="${API_BASE%/}"
    ask "Timeout(秒)" "$TIMEOUT"; TIMEOUT="$REPLY"
    [ -n "$API_BASE" ] || { echo -e "  ${RED}API Base 不能为空。${RESET}" >&2; return 1; }
    [ -n "$MODEL" ] || { echo -e "  ${RED}Model 不能为空。${RESET}" >&2; return 1; }
    save_profile_json "$file" "$PROVIDER" "$API_BASE" "$API_KEY" "$MODEL" "$TIMEOUT"
    printf '%s\n' "$profile_name" > "$LLM_CURRENT_FILE"
    echo -e "  ${GREEN}已设为当前配置: $profile_name${RESET}"
}

delete_action() {
    if ! choose_llm_profile; then echo -e "  ${YELLOW}没有可删除的配置。${RESET}"; return 0; fi
    rm -f "$SELECTED_PROFILE_FILE"
    if [ -f "$LLM_CURRENT_FILE" ] && [ "$(cat "$LLM_CURRENT_FILE" 2>/dev/null || true)" = "$SELECTED_PROFILE" ]; then
        rm -f "$LLM_CURRENT_FILE"
    fi
    echo -e "  ${GREEN}已删除: $SELECTED_PROFILE${RESET}"
}

test_action() {
    if ! choose_llm_profile; then echo -e "  ${YELLOW}请先创建配置。${RESET}"; return 0; fi
    load_llm_profile "$SELECTED_PROFILE"
    echo ""
    echo -e "  配置 : $SELECTED_PROFILE"
    echo -e "  API  : $API_BASE"
    echo -e "  Model: $MODEL"
    local args=(-fsS --max-time 10)
    [ -n "$API_KEY" ] && args+=(-H "Authorization: Bearer $API_KEY")
    echo -e "  正在访问: $API_BASE/models"
    if curl "${args[@]}" "$API_BASE/models" > /tmp/itool_llm_test_response 2>/tmp/itool_llm_test_error; then
        echo -e "  ${GREEN}连接成功。${RESET}"
        head -c 300 /tmp/itool_llm_test_response 2>/dev/null || true
        echo ""
    else
        echo -e "  ${RED}连接失败或该服务不支持 /models。${RESET}"
        cat /tmp/itool_llm_test_error 2>/dev/null || true
    fi
    rm -f /tmp/itool_llm_test_response /tmp/itool_llm_test_error
}

while true; do
    echo ""
    echo -e "  ${WHITE}===== 大模型配置管理 =====${RESET}"
    print_config_list
    echo ""
    echo -e "    ${GREEN}A${RESET}) 将某个配置设为当前使用"
    echo -e "    ${GREEN}B${RESET}) 新建配置"
    echo -e "    ${GREEN}C${RESET}) 修改配置"
    echo -e "    ${GREEN}D${RESET}) 删除配置"
    echo -e "    ${GREEN}E${RESET}) 测试配置"
    echo -e "    ${GREEN}Q${RESET}) 退出"
    ask "请选择" "Q"
    case "$REPLY" in
        a|A|1)
            if choose_llm_profile; then
                printf '%s\n' "$SELECTED_PROFILE" > "$LLM_CURRENT_FILE"
                echo -e "  ${GREEN}当前配置已设为: $SELECTED_PROFILE${RESET}"
            else
                echo -e "  ${YELLOW}请先新建配置。${RESET}"
            fi
            ;;
        b|B|2) create_or_edit_action new ;;
        c|C|3) create_or_edit_action edit ;;
        d|D|4) delete_action ;;
        e|E|5) test_action ;;
        q|Q|6) echo -e "  ${GREEN}退出。${RESET}"; exit 0 ;;
        *) echo -e "  ${YELLOW}无效选择。${RESET}" ;;
    esac
done
