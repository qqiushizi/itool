#!/bin/bash
# ============================================================
# 算子设计 / 需求分析 (支持大模型分析与手动填写)
#
# 模式:
#   [1] 本地大模型: 昇腾宿主机/容器内 vLLM-ascend 服务
#   [2] 外部 API:   OpenAI Chat Completions 兼容接口
#   [3] 手动填写:   保留原交互式人工定义流程
#
# 大模型模式环境变量:
#   OP_SPEC_MODE=llm|manual        # 跳过分析方式选择
#   LLM_PROVIDER=local|external    # OP_SPEC_MODE=llm 时, 跳过接入方式选择
#   OP_DESC_LLM="..."              # 大模型模式: 算子需求描述
#   ITOOL_LLM_API_BASE=...         # 本地默认 http://127.0.0.1:8000/v1; 外部默认 https://api.deepseek.com/v1
#   ITOOL_LLM_API_KEY=...          # 外部 API 使用; 宿主机 vLLM 服务可留空
#   ITOOL_LLM_MODEL=...            # vLLM served model name 或外部模型名
#   ITOOL_LLM_TIMEOUT=...          # 默认 120s
#
# 输出:
#   <out>/op.json      (msopgen 工程生成用)
#   <out>/op_spec.md   (可读的需求规格说明)
# ============================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

read_def() {  # $1=提示 $2=默认值 ; 结果放 $REPLY
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
    return 0
}
read_secret() {  # $1=提示 $2=默认值 ; 结果放 $REPLY, 不回显
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    stty -echo 2>/dev/null || true
    IFS= read -r REPLY || REPLY=""
    stty echo 2>/dev/null || true
    echo ""
    [ -z "$REPLY" ] && REPLY="$def"
    return 0
}
confirm() {
    local ans
    printf "  %s [y/N]: " "$1"
    IFS= read -r ans || ans=""
    case "$ans" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

manual_flow() {
    echo ""
    echo -e "  ${CYAN}===== 算子设计 / 需求分析 (手动填写) =====${RESET}"
    echo -e "  ${YELLOW}(回车使用默认值; 输入以逗号分隔多值)${RESET}"
    echo ""

    # ---- 基本信息 ----
    read_def "算子名称" "AddCustom";        OP_NAME="$REPLY"
    read_def "算子类型(elementwise/matmul/reduce/custom)" "custom"; OP_TYPE="$REPLY"
    printf "  算子功能描述: "; IFS= read -r OP_DESC || OP_DESC=""; OP_DESC="${OP_DESC:-未提供}"
    read_def "支持数据类型(如 fp16,fp32,int8)" "fp16,fp32"; OP_DTYPES="$REPLY"

    # ---- 输入 ----
    read_def "输入数量" "2"; N_IN="$REPLY"
    IN_NAMES=(); IN_DTYPES=(); IN_SHAPES=()
    for ((i=1; i<=N_IN; i++)); do
        echo -e "  ${CYAN}--- 输入 $i ---${RESET}"
        read_def "    名称" "x$i";                     IN_NAMES+=("$REPLY")
        read_def "    数据类型(如 fp16)" "fp16";       IN_DTYPES+=("$REPLY")
        read_def "    典型shape(如 1024,1024; -1 表动态)" "-1,-1"; IN_SHAPES+=("$REPLY")
    done

    # ---- 输出 ----
    read_def "输出数量" "1"; N_OUT="$REPLY"
    OUT_NAMES=(); OUT_DTYPES=(); OUT_SHAPES=()
    for ((i=1; i<=N_OUT; i++)); do
        echo -e "  ${CYAN}--- 输出 $i ---${RESET}"
        read_def "    名称" "y$i";                     OUT_NAMES+=("$REPLY")
        read_def "    数据类型(如 fp16)" "fp16";       OUT_DTYPES+=("$REPLY")
        read_def "    典型shape(如 1024,1024)" "-1,-1"; OUT_SHAPES+=("$REPLY")
    done

    # ---- 属性(可选) ----
    read_def "属性(可选, 格式 name:type:value, 逗号分隔, 如 alpha:float:1.0)" ""; OP_ATTRS="$REPLY"

    # ---- 输出目录 ----
    OUT_DIR="${OUT_DIR:-./op_design_${OP_NAME}}"
    mkdir -p "$OUT_DIR"

    # 把多值数据写成 TSV, 交给 python 生成合法 JSON(避免引号转义问题)
    IN_TSV="$OUT_DIR/.inputs.tsv";  : > "$IN_TSV"
    for ((i=0; i<N_IN; i++)); do printf '%s\t%s\t%s\n' "${IN_NAMES[$i]}" "${IN_DTYPES[$i]}" "${IN_SHAPES[$i]}" >> "$IN_TSV"; done
    OUT_TSV="$OUT_DIR/.outputs.tsv"; : > "$OUT_TSV"
    for ((i=0; i<N_OUT; i++)); do printf '%s\t%s\t%s\n' "${OUT_NAMES[$i]}" "${OUT_DTYPES[$i]}" "${OUT_SHAPES[$i]}" >> "$OUT_TSV"; done

    OP_NAME="$OP_NAME" OP_TYPE="$OP_TYPE" OP_DESC="$OP_DESC" OP_DTYPES="$OP_DTYPES" OP_ATTRS="$OP_ATTRS" \
    IN_TSV="$IN_TSV" OUT_TSV="$OUT_TSV" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os, json

def split_csv(s):
    return [x.strip() for x in s.split(',') if x.strip()]

def read_tsv(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            name, dtype, shape = line.split('\t')
            items.append({
                "name": name,
                "param_type": "required",
                "format": ["ND"],
                "type": split_csv(dtype),
                "shape": split_csv(shape),
            })
    return items

def parse_attrs(s):
    attrs = []
    for part in split_csv(s):
        bits = part.split(':')
        if len(bits) == 3:
            attrs.append({"name": bits[0].strip(), "type": bits[1].strip(), "value": bits[2].strip()})
        elif len(bits) == 2:
            attrs.append({"name": bits[0].strip(), "type": bits[1].strip(), "value": ""})
    return attrs

op = {
    "op": os.environ['OP_NAME'],
    "language": "cpp",
    "input_desc": read_tsv(os.environ['IN_TSV']),
    "output_desc": read_tsv(os.environ['OUT_TSV']),
    "attr": parse_attrs(os.environ.get('OP_ATTRS', '')),
}

out_dir = os.environ['OUT_DIR']
with open(os.path.join(out_dir, 'op.json'), 'w', encoding='utf-8') as f:
    json.dump([op], f, ensure_ascii=False, indent=2)

lines = []
lines.append("# 算子需求规格: %s" % op['op'])
lines.append("")
lines.append("- 算子类型: %s" % os.environ['OP_TYPE'])
lines.append("- 功能描述: %s" % os.environ['OP_DESC'])
lines.append("- 支持数据类型: %s" % os.environ['OP_DTYPES'])
lines.append("")
lines.append("## 输入")
for d in op['input_desc']:
    lines.append("- `%s`: type=%s shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
lines.append("")
lines.append("## 输出")
for d in op['output_desc']:
    lines.append("- `%s`: type=%s shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
lines.append("")
lines.append("## 属性")
if op['attr']:
    for a in op['attr']:
        lines.append("- `%s`: %s = %s" % (a['name'], a['type'], a.get('value', '')))
else:
    lines.append("- (无)")
lines.append("")
with open(os.path.join(out_dir, 'op_spec.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print("")
print("生成的 op.json : %s/op.json" % out_dir)
print("生成的 op_spec.md: %s/op_spec.md" % out_dir)
PY

    # 清理临时文件
    rm -f "$IN_TSV" "$OUT_TSV"

    echo ""
    echo -e "  ${GREEN}✔ 已生成:${RESET}"
    echo -e "    $OUT_DIR/op.json"
    echo -e "    $OUT_DIR/op_spec.md"
    echo ""
    echo "下一步生成工程:"
    echo "  bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh $OUT_DIR/op.json"
}

# ---------- 主流程 ----------
echo ""
echo -e "  ${CYAN}===== 算子设计 / 需求分析 =====${RESET}"
echo ""

MODE="${OP_SPEC_MODE:-}"
PROVIDER="${LLM_PROVIDER:-}"

if [ -z "$MODE" ]; then
    echo -e "  ${WHITE}请选择分析方式:${RESET}"
    echo "    [1] 本地大模型 (昇腾宿主机/容器内 vLLM-ascend 服务)"
    echo "    [2] 外部 API (OpenAI Chat Completions 兼容接口)"
    echo "    [3] 手动填写"
    printf "  请选择 [1]: "
    IFS= read -r ans || ans=""
    ans="${ans:-1}"
    case "$ans" in
        2) MODE="llm"; PROVIDER="external" ;;
        3) MODE="manual" ;;
        *) MODE="llm"; PROVIDER="local" ;;
    esac
elif [ "$MODE" = "llm" ]; then
    if [ -z "$PROVIDER" ]; then
        echo -e "  ${WHITE}请选择大模型接入方式:${RESET}"
        echo "    [1] 本地大模型 (昇腾宿主机/容器内 vLLM-ascend 服务)"
        echo "    [2] 外部 API (OpenAI Chat Completions 兼容接口)"
        printf "  请选择 [1]: "
        IFS= read -r ans || ans=""
        ans="${ans:-1}"
        case "$ans" in
            2) PROVIDER="external" ;;
            *) PROVIDER="local" ;;
        esac
    fi
fi

if [ "$MODE" = "manual" ]; then
    manual_flow
    exit $?
fi

# ============================================================
# 大模型分析模式: 先按接入方式声明变量, 再收集算子需求
# ============================================================
API_BASE="${ITOOL_LLM_API_BASE:-}"
API_KEY="${ITOOL_LLM_API_KEY:-}"
MODEL="${ITOOL_LLM_MODEL:-}"
TIMEOUT="${ITOOL_LLM_TIMEOUT:-120}"

echo ""
if [ "$PROVIDER" = "external" ]; then
    echo -e "  ${CYAN}===== 外部 API 变量声明 =====${RESET}"
    if [ -z "$API_BASE" ]; then
        read_def "ITOOL_LLM_API_BASE" "https://api.deepseek.com/v1"
        API_BASE="$REPLY"
    fi
    API_BASE="${API_BASE%/}"
    if [ -z "$MODEL" ]; then
        read_def "ITOOL_LLM_MODEL" "deepseek-chat"
        MODEL="$REPLY"
    fi
    if [ -z "$API_KEY" ]; then
        read_secret "ITOOL_LLM_API_KEY" ""
        API_KEY="$REPLY"
    fi
else
    echo -e "  ${CYAN}===== 本地大模型变量声明 =====${RESET}"
    echo -e "  ${YELLOW}目标: 昇腾宿主机/容器内已启动的 vLLM-ascend 服务${RESET}"
    if [ -z "$API_BASE" ]; then
        read_def "ITOOL_LLM_API_BASE" "http://127.0.0.1:8000/v1"
        API_BASE="$REPLY"
    fi
    API_BASE="${API_BASE%/}"
    if [ -z "$MODEL" ]; then
        read_def "ITOOL_LLM_MODEL(vLLM served model name)" "Qwen/Qwen2.5-7B-Instruct"
        MODEL="$REPLY"
    fi
    if [ -z "$API_KEY" ]; then
        read_secret "ITOOL_LLM_API_KEY(本地服务可留空)" ""
        API_KEY="$REPLY"
    fi
fi

[ -n "$API_BASE" ] || { echo -e "${RED}API Base 不能为空。${RESET}" >&2; exit 1; }
[ -n "$MODEL" ] || { echo -e "${RED}模型名不能为空。${RESET}" >&2; exit 1; }

DESC="${OP_DESC_LLM:-}"
if [ -z "$DESC" ]; then
    echo ""
    read_def "算子需求描述(自然语言, 建议说明算子名/输入输出/shape/数据类型)" ""
    DESC="$REPLY"
fi
[ -n "$DESC" ] || { echo -e "${RED}算子需求描述不能为空。${RESET}" >&2; exit 1; }

echo ""
echo -e "  ${CYAN}本次大模型配置:${RESET}"
echo "    ITOOL_LLM_API_BASE = $API_BASE"
echo "    ITOOL_LLM_MODEL    = $MODEL"
echo "    ITOOL_LLM_TIMEOUT  = ${TIMEOUT}s"
if [ -n "$API_KEY" ]; then
    echo "    ITOOL_LLM_API_KEY  = (已设置, 不显示明文)"
else
    echo "    ITOOL_LLM_API_KEY  = (未设置)"
fi

TMP_DIR=$(mktemp -d /tmp/itool-op-spec.XXXXXX)
trap 'rm -rf "$TMP_DIR"' EXIT
PAYLOAD="$TMP_DIR/payload.json"
RESP="$TMP_DIR/response.json"
PARSED="$TMP_DIR/parsed.json"
OP_NAME_FILE="$TMP_DIR/op_name"

# 构建 Chat Completions 请求体
DESC="$DESC" MODEL="$MODEL" PAYLOAD="$PAYLOAD" python3 - <<'PY'
import os, json
with open(os.environ['PAYLOAD'], 'w', encoding='utf-8') as f:
    json.dump({
        "model": os.environ['MODEL'],
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一个昇腾算子需求分析助手。请从用户描述中提取算子结构化定义，并只输出 JSON，不要输出解释。"
                    "JSON 结构必须如下，并确保 output_desc 的 name 不与 input_desc 冲突：\n"
                    '[\n'
                    '  {\n'
                    '    "op": "算子名称",\n'
                    '    "language": "cpp",\n'
                    '    "input_desc": [\n'
                    '      {"name":"A","param_type":"required","format":["ND"],"type":["fp16"],"shape":["-1","1024"]}\n'
                    '    ],\n'
                    '    "output_desc": [\n'
                    '      {"name":"C","param_type":"required","format":["ND"],"type":["fp16"],"shape":["-1","1024"]}\n'
                    '    ],\n'
                    '    "attr": []\n'
                    '  }\n'
                    ']\n'
                    "type 只能是常见类型，如 fp16/fp32/int8/int32/int64/bool。"
                    "shape 元素用字符串表达；动态维用 -1。"
                ),
            },
            {"role": "user", "content": os.environ['DESC']},
        ],
    }, f, ensure_ascii=False)
PY

curl_args=(-sS --max-time "$TIMEOUT" -H "Content-Type: application/json")
[ -n "$API_KEY" ] && curl_args+=(-H "Authorization: Bearer $API_KEY")

echo ""
echo -e "  ${CYAN}调用大模型: $API_BASE/chat/completions${RESET}"

if curl "${curl_args[@]}" -o "$RESP" -d @"$PAYLOAD" "$API_BASE/chat/completions"; then
    :
else
    curl_rc=$?
    echo -e "${RED}大模型调用失败(退出码 $curl_rc)。${RESET}" >&2
    echo "  API Base: $API_BASE"
    echo "  Model   : $MODEL"
    echo "  可手动检查服务是否可达。" >&2
    exit 1
fi

# 解析并校验模型输出；通过则打印摘要并写入临时解析文件
RESP="$RESP" PARSED="$PARSED" OP_NAME_FILE="$OP_NAME_FILE" python3 - <<'PY'
import os, re, json, sys

resp_path = os.environ['RESP']
parsed_path = os.environ['PARSED']
op_name_path = os.environ['OP_NAME_FILE']

try:
    data = json.load(open(resp_path, 'r', encoding='utf-8'))
except Exception as e:
    print("大模型返回不是合法 JSON: %s" % e)
    print("原始返回片段: %s" % open(resp_path, 'r', encoding='utf-8').read()[:500])
    raise

content = None
try:
    content = data['choices'][0]['message']['content']
except Exception:
    if isinstance(data, dict) and data.get('error'):
        print("API 返回错误: %s" % data['error'])
    else:
        print("无法从返回中提取 choices[0].message.content")
    raise

def extract_json(text):
    text = text.strip()
    fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.S)
    if fenced:
        return fenced.group(1).strip()
    start = text.find('[')
    if start < 0:
        start = text.find('{')
    if start < 0:
        return text
    return text[start:]

extracted = extract_json(content)
try:
    obj = json.loads(extracted)
except Exception as e:
    # 尝试从第一个 { 到最后一个 } / 第一个 [ 到最后一个 ]
    if extracted.lstrip().startswith('['):
        obj = json.loads(extracted[:extracted.rfind(']')+1])
    else:
        obj = json.loads(extracted[:extracted.rfind('}')+1])

if isinstance(obj, dict):
    ops = [obj]
else:
    ops = obj
if not isinstance(ops, list) or not all(isinstance(op, dict) for op in ops):
    raise ValueError('模型输出结构异常, 预期为 JSON 数组')

op = ops[0]
if 'op' not in op or not isinstance(op.get('op'), str):
    raise ValueError('缺少字符串字段 op')
op['language'] = 'cpp'

def norm_desc(items, key):
    if key not in op:
        op[key] = []
    if not isinstance(op[key], list):
        raise ValueError('%s 必须是数组' % key)
    out=[]
    for item in op[key]:
        if not isinstance(item, dict):
            raise ValueError('%s 元素必须是对象' % key)
        types = item.get('type') or item.get('dtype') or []
        if isinstance(types, str):
            types = [types]
        shape = item.get('shape') or item.get('shapes') or []
        if isinstance(shape, str):
            shape = [shape]
        out.append({
            'name': str(item.get('name', '')),
            'param_type': str(item.get('param_type', 'required')),
            'format': item.get('format') or ['ND'],
            'type': [str(x) for x in types],
            'shape': [str(x) for x in shape],
        })
        if not out[-1]['name']:
            raise ValueError('%s 中存在缺少 name 的项' % key)
        if not out[-1]['type']:
            out[-1]['type'] = ['fp16']
        if not out[-1]['shape']:
            out[-1]['shape'] = ['-1']
    op[key] = out

norm_desc(op, 'input_desc')
norm_desc(op, 'output_desc')

if 'attr' not in op:
    op['attr'] = []
if not isinstance(op['attr'], list):
    raise ValueError('attr 必须是数组')

with open(parsed_path, 'w', encoding='utf-8') as f:
    json.dump(op, f, ensure_ascii=False, indent=2)
with open(op_name_path, 'w', encoding='utf-8') as f:
    f.write(op['op'])

print("")
print("模型解析结果:")
print("- 算子名称: %s" % op['op'])
print("- 语言     : %s" % op['language'])
print("- 输入     :")
for d in op['input_desc']:
    print("    %s  type=%s  shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
print("- 输出     :")
for d in op['output_desc']:
    print("    %s  type=%s  shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
print("- attr     :")
if op['attr']:
    for a in op['attr']:
        print("    %s=%s (%s)" % (a.get('name'), a.get('value'), a.get('type')))
else:
    print("    (无)")
PY

OP_NAME=$(cat "$OP_NAME_FILE")
OUT_DIR="${OUT_DIR:-./op_design_${OP_NAME}}"

if ! confirm "是否使用以上解析结果生成 op.json 和 op_spec.md?"; then
    if confirm "是否改为手动填写?"; then
        manual_flow
        exit $?
    fi
    echo -e "${YELLOW}已取消。${RESET}"
    exit 0
fi

mkdir -p "$OUT_DIR"
PARSED="$PARSED" OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os, json
op = json.load(open(os.environ['PARSED'], 'r', encoding='utf-8'))
out_dir = os.environ['OUT_DIR']

with open(os.path.join(out_dir, 'op.json'), 'w', encoding='utf-8') as f:
    json.dump([op], f, ensure_ascii=False, indent=2)

lines=[]
lines.append("# 算子需求规格: %s" % op['op'])
lines.append("")
lines.append("- 语言: cpp")
lines.append("")
lines.append("## 输入")
for d in op['input_desc']:
    lines.append("- `%s`: type=%s shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
lines.append("")
lines.append("## 输出")
for d in op['output_desc']:
    lines.append("- `%s`: type=%s shape=%s" % (d['name'], ','.join(d['type']), ','.join(d['shape'])))
lines.append("")
lines.append("## 属性")
if op['attr']:
    for a in op['attr']:
        lines.append("- `%s`: %s = %s" % (a.get('name',''), a.get('type',''), a.get('value','')))
else:
    lines.append("- (无)")
lines.append("")
with open(os.path.join(out_dir, 'op_spec.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines)+'\n')
PY

echo ""
echo -e "  ${GREEN}✔ 已生成:${RESET}"
echo -e "    $OUT_DIR/op.json"
echo -e "    $OUT_DIR/op_spec.md"
echo ""
echo "下一步生成工程:"
echo "  bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh $OUT_DIR/op.json"
