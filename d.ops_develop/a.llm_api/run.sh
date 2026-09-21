#!/bin/bash
# ============================================================
# ① 配置 OpenCode 的 API（OpenAI 兼容）
#
# 功能:
#   交互式录入 OpenAI 兼容接口的 baseURL / apiKey / model / provider,
#   写入 opencode 的配置文件 opencode.json, 供其调用大模型。
#
# 用法:
#   bash a.llm_api/run.sh                    # 交互式纵向菜单
#   bash a.llm_api/run.sh config             # 直接进入「配置 API」
#   bash a.llm_api/run.sh view               # 只打印当前配置
#   bash a.llm_api/run.sh test               # 只测试连通性
#
# 非交互环境变量(有值则不再交互询问):
#   OPENCODE_API_BASE    OpenAI 兼容 baseURL
#   OPENCODE_API_KEY     API Key(本地服务可留空)
#   OPENCODE_MODEL       模型名
#   OPENCODE_PROVIDER    provider 名, 默认 openai-compatible
#   OPENCODE_TAG         provider 名, 默认 openai-compatible
#   OPENCODE_TIMEOUT     超时(毫秒), 默认 120000
#   OPENCODE_CONFIG      配置文件路径, 默认 ~/.config/opencode/opencode.json
# ============================================================
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")

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

# ---------- 定位 opencode 配置文件 ----------
resolve_config_file() {
    if [ -n "${OPENCODE_CONFIG:-}" ]; then
        CONFIG_FILE="$OPENCODE_CONFIG"
        return 0
    fi
    # 绿色包便携版 opencode 用 XDG_CONFIG_HOME 指向 config/，默认仍选标准位置
    CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json"
}

# ---------- 从已有 opencode.json 里读出默认值 ----------
# 结果写 PROVIDER_V/BASE_V/MODEL_V/TIMEOUT_V/KEY_V
load_existing() {
    PROVIDER_V="openai-compatible"; BASE_V=""; MODEL_V=""; TIMEOUT_V="120000"; KEY_V=""
    [ -f "$CONFIG_FILE" ] || return 0
    python3 - "$CONFIG_FILE" <<'PY' 2>/dev/null || return 0
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p, encoding='utf-8'))
except Exception:
    sys.exit(0)
providers = d.get('provider') or {}
# 优先取 model 里 "provider/model" 的 provider
pro = 'openai-compatible'
model = d.get('model') or ''
if isinstance(model, str) and '/' in model:
    pro = model.split('/', 1)[0]
elif providers:
    pro = next(iter(providers))
c = providers.get(pro) or {}
opts = c.get('options') or {}
models = c.get('models') or {}
mid = ''
if isinstance(model, str) and '/' in model:
    mid = model.split('/', 1)[1]
if not mid and models:
    mid = next(iter(models))
base = opts.get('baseURL') or ''
timeout = opts.get('timeout') or '120000'
key = opts.get('apiKey') or ''
print('PROVIDER=%s' % pro)
print('BASE=%s' % base)
print('MODEL=%s' % mid)
print('TIMEOUT=%s' % timeout)
print('KEY=%s' % key)
PY
}

# ---------- 写入 opencode.json ----------
write_config() {
    local provider="$1" base="$2" key="$3" model="$4" timeout="$5" file="$6"
    CONFIG_FILE="$file" PROVIDER="$provider" BASE="$base" KEY="$key" MODEL="$model" TIMEOUT="$timeout" \
    python3 - <<'PY'
import os, json
def clean(v):
    return str(v).encode('utf-8', 'replace').decode('utf-8')
file = clean(os.environ['CONFIG_FILE'])
provider = clean(os.environ['PROVIDER'])
base = clean(os.environ['BASE']).rstrip('/')
key = clean(os.environ['KEY'])
model = clean(os.environ['MODEL'])
try:
    timeout = int(os.environ['TIMEOUT'] or '120000')
except ValueError:
    timeout = 120000

try:
    with open(file, encoding='utf-8') as f:
        c = json.load(f)
except Exception:
    c = {}
if not isinstance(c, dict):
    c = {}

c.setdefault('$schema', 'https://opencode.ai/config.json')
c.setdefault('provider', {})
c.setdefault('tools', {"bash": True, "edit": True, "read": True, "write": True})
c.setdefault('autoupdate', False)
c['provider'][provider] = {
    'npm': '@ai-sdk/openai-compatible',
    'name': provider,
    'options': {'baseURL': base, 'timeout': timeout, 'apiKey': key},
    'models': {model: {'name': model}}
}
c['model'] = '%s/%s' % (provider, model)

os.makedirs(os.path.dirname(file) or '.', exist_ok=True)
with open(file, 'w', encoding='utf-8') as f:
    json.dump(c, f, ensure_ascii=False, indent=2)
print('已写入: %s' % file)
PY
}

# ---------- 交互式配置 API ----------
do_config() {
    resolve_config_file
    load_existing
    local provider base key model timeout
    provider="${OPENCODE_PROVIDER:-${OPENCODE_TAG:-$PROVIDER_V}}"
    base="${OPENCODE_API_BASE:-$BASE_V}"
    key="${OPENCODE_API_KEY:-$KEY_V}"
    model="${OPENCODE_MODEL:-$MODEL_V}"
    timeout="${OPENCODE_TIMEOUT:-$TIMEOUT_V}"

    echo ""
    echo -e "  ${CYAN}配置 OpenCode 的 API${RESET}"
    echo -e "  配置文件: ${CONFIG_FILE}"
    echo ""
    if [ -n "${OPENCODE_PROVIDER:-}${OPENCODE_TAG:-}" ]; then provider="${OPENCODE_PROVIDER:-$OPENCODE_TAG}"; else
        ask "Provider 名" "$provider"; provider="$REPLY"
    fi
    if [ -n "${OPENCODE_API_BASE:-}" ]; then :; else
        ask "API Base URL" "${base:-https://api.deepseek.com/v1}"; base="$REPLY"
    fi
    if [ -n "${OPENCODE_API_KEY:-}" ]; then :; else
        read_secret "API Key(本地可留空)" "$key"; key="$REPLY"
    fi
    if [ -n "${OPENCODE_MODEL:-}" ]; then :; else
        ask "Model" "${model:-deepseek-chat}"; model="$REPLY"
    fi
    if [ -n "${OPENCODE_TIMEOUT:-}" ]; then :; else
        ask "Timeout(毫秒)" "$timeout"; timeout="$REPLY"
    fi

    base="${base%/}"
    [ -n "$base" ] || { echo -e "  ${RED}baseURL 不能为空。${RESET}" >&2; return 1; }
    [ -n "$model" ] || { echo -e "  ${RED}model 不能为空。${RESET}" >&2; return 1; }
    [ -n "$provider" ] || provider="openai-compatible"

    write_config "$provider" "$base" "$key" "$model" "$timeout" "$CONFIG_FILE"
    echo -e "  ${GREEN}配置完成:  model = $provider/$model${RESET}"
}

# ---------- 查看当前配置 ----------
do_view() {
    resolve_config_file
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "  ${YELLOW}尚未配置(配置文件不存在): $CONFIG_FILE${RESET}"
        return 0
    fi
    echo -e "  配置文件: $CONFIG_FILE"
    sed 's/^/  /' "$CONFIG_FILE" 2>/dev/null || cat "$CONFIG_FILE"
}

# ---------- 测试连接 ----------
do_test() {
    resolve_config_file
    [ -f "$CONFIG_FILE" ] || { echo -e "  ${YELLOW}尚未配置, 请先配置。${RESET}"; return 0; }
    load_existing
    local base="$BASE_V" key="$KEY_V"
    [ -n "$base" ] || { echo -e "  ${YELLOW}配置里没有 baseURL。${RESET}"; return 0; }
    echo ""
    echo -e "  配置 : $CONFIG_FILE"
    echo -e "  API  : $base"
    local args=(-fsS --max-time 10)
    [ -n "$key" ] && args+=(-H "Authorization: Bearer $key")
    echo -e "  正在访问: $base/models"
    if curl "${args[@]}" "$base/models" > /tmp/itool_oc_test 2>/tmp/itool_oc_test_err; then
        echo -e "  ${GREEN}连接成功。${RESET}"
        head -c 300 /tmp/itool_oc_test 2>/dev/null || true
        echo ""
    else
        echo -e "  ${RED}连接失败或该服务不支持 /models。${RESET}"
        tail -n 3 /tmp/itool_oc_test_err 2>/dev/null || true
    fi
    rm -f /tmp/itool_oc_test /tmp/itool_oc_test_err
}

# ---------- 主流程 ----------
ACTION="${1:-menu}"
case "$ACTION" in
    config) do_config; exit $? ;;
    view) do_view; exit $? ;;
    test) do_test; exit $? ;;
esac

while true; do
    echo ""
    echo -e "  ${WHITE}===== 配置 OpenCode 的 API =====${RESET}"
    resolve_config_file
    echo -e "  配置文件: ${CYAN}$CONFIG_FILE${RESET}"
    echo ""
    echo -e "    ${GREEN}A${RESET}) 配置 API(baseURL / apiKey / model / provider)"
    echo -e "    ${GREEN}B${RESET}) 查看当前配置"
    echo -e "    ${GREEN}C${RESET}) 测试连接(访问 baseURL/models)"
    echo -e "    ${GREEN}Q${RESET}) 退出"
    ask "请选择" "A"
    case "$REPLY" in
        a|A|1) do_config ;;
        b|B|2) do_view ;;
        c|C|3) do_test ;;
        q|Q|4) echo -e "  ${GREEN}退出。${RESET}"; exit 0 ;;
        *) echo -e "  ${YELLOW}无效选择。${RESET}" ;;
    esac
done
