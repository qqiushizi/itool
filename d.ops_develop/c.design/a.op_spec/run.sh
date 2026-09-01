#!/bin/bash
# ============================================================
# 算子设计 / 需求分析
# 用户提供: 算子功能说明、支持的数据类型、典型 shape、算子功能是什么
# 产出:   <out>/op.json      (msopgen 工程生成用)
#         <out>/op_spec.md   (可读的需求规格说明)
# ============================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RESET='\033[0m'

read_def() {  # $1=提示 $2=默认值 ; 结果放 $REPLY
    local prompt="$1" def="$2"
    printf "  %s [%s]: " "$prompt" "$def"
    IFS= read -r REPLY || REPLY=""
    [ -z "$REPLY" ] && REPLY="$def"
    return 0
}

echo ""
echo -e "  ${CYAN}===== 算子设计 / 需求分析 =====${RESET}"
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
import os, json, sys

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

# 可读规格
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

print("op.json : %s/op.json" % out_dir)
print("op_spec.md: %s/op_spec.md" % out_dir)
PY

# 清理临时文件
rm -f "$IN_TSV" "$OUT_TSV"

echo ""
echo -e "  ${GREEN}✔ 已生成:${RESET}"
echo -e "    $OUT_DIR/op.json"
echo -e "    $OUT_DIR/op_spec.md"
echo ""
echo "下一步生成工程:"
echo "  bash d.ops_develop/d.scaffold/a.msopgen/run.sh $OUT_DIR/op.json"
