#!/bin/bash
# ============================================================
# ⑦ 大模型辅助修改算子工程代码
# 用法:
#   bash d.ops_develop/g.op_fix/run.sh [op_build目录]
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
resolve_ops_root "$SCRIPT_DIR"

PROJECT_IN="${1:-}"
if [ -n "$PROJECT_IN" ]; then
    PROJECT_DIR="$PROJECT_IN"
    [ -d "$PROJECT_DIR" ] || { echo -e "${RED}目录不存在: $PROJECT_DIR${RESET}" >&2; exit 1; }
else
    PROJECTS=()
    for d in "$OPS_WORKSPACE"/op_build_*; do
        [ -d "$d" ] || continue
        PROJECTS+=("$d")
    done
    if [ ${#PROJECTS[@]} -eq 0 ]; then
        echo -e "${RED}没有找到算子工程，请先运行 f.op_build。${RESET}" >&2
        exit 1
    fi
    echo -e "  ${CYAN}请选择要修改的算子工程:${RESET}"
    for ((i=0; i<${#PROJECTS[@]}; i++)); do
        printf '    %3d) %s\n' "$((i+1))" "$(basename "${PROJECTS[$i]}")"
    done
    while true; do
        ask "请选择工程编号" "1"
        n="$REPLY"
        if [ "$n" -ge 1 ] 2>/dev/null && [ "$n" -le "${#PROJECTS[@]}" ] 2>/dev/null; then
            PROJECT_DIR="${PROJECTS[$((n-1))]}"
            break
        fi
        echo -e "  ${RED}编号无效。${RESET}"
    done
fi

if ! choose_llm_profile; then
    echo -e "${RED}未找到大模型配置，请先运行 a.llm_config。${RESET}" >&2
    exit 1
fi
load_llm_profile "$SELECTED_PROFILE"

echo ""
echo -e "  ${WHITE}===== 算子工程代码辅助修改 =====${RESET}"
printf '  %-16s: %s\n' "工程" "$PROJECT_DIR"
printf '  %-16s: %s\n' "模型配置" "$SELECTED_PROFILE"
printf '  %-16s: %s\n' "API" "$API_BASE"
printf '  %-16s: %s\n' "Model" "$MODEL"
echo ""

OP_FIX_PY=$(mktemp "${TMPDIR:-/tmp}/itool-op-fix.XXXXXX.py")
trap 'rm -f "$OP_FIX_PY"' EXIT
cat > "$OP_FIX_PY" <<'PY'
import os, sys, re, json, pathlib, urllib.request, urllib.error, shutil, time

project_dir = pathlib.Path(os.environ['OP_FIX_PROJECT']).resolve()
api_base = os.environ['OP_FIX_API_BASE'].rstrip('/')
api_key = os.environ['OP_FIX_API_KEY']
model = os.environ['OP_FIX_MODEL']
timeout = int(os.environ.get('OP_FIX_TIMEOUT') or 120)
hist_dir = project_dir / '.itool'
hist_dir.mkdir(exist_ok=True)
hist_path = hist_dir / 'op_fix_history.jsonl'

SYSTEM_TEMPLATE = """你是一个昇腾 CANN 算子工程代码辅助助手。用户会要求检查或修改当前算子工程。
要求：
1) 如需修改文件，每个文件单独用以下格式输出：
<<<FILE:相对路径>>>
完整新文件内容
<<<END>>>
2) 不要输出不存在的文件，不要输出解释。
3) 相对路径必须基于工程目录 {proj}。
当前工程目录: {proj}
"""

FILE_EXTS = {'.cpp','.h','.hpp','.py','.sh','.txt','.md','.cmake'}
def text_files(project):
    files=[]
    for p in sorted(project.rglob('*')):
        if not p.is_file(): continue
        if any(part in {'.git','build_out','.itool'} for part in p.parts): continue
        if p.suffix.lower() not in FILE_EXTS and p.name not in {'CMakeLists.txt','build.sh','CMakePresets.json','Makefile'}: continue
        try:
            txt=p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        files.append((p, txt))
    return files

def system_prompt():
    files=text_files(project_dir)
    tree=[]
    total=0
    for p,txt in files:
        rel=p.relative_to(project_dir)
        tree.append(str(rel))
        total += len(txt)
    header=SYSTEM_TEMPLATE.format(proj=project_dir)
    header += '工程文件列表:\n' + '\n'.join('  '+t for t in tree) + '\n'
    if total <= 60000:
        header += '\n以下是当前工程文件内容:\n'
        for p,txt in files:
            rel=p.relative_to(project_dir)
            header += '\n===== %s =====\n%s\n' % (rel, txt)
    else:
        header += '\n工程文件较大，如需查看具体文件内容，请在对话中明确要求。\n'
    return header

def load_history():
    if not hist_path.exists():
        return []
    msgs=[]
    with open(hist_path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                try: msgs.append(json.loads(line))
                except Exception: pass
    return msgs

def append_history(msg):
    with open(hist_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(msg, ensure_ascii=False)+'\n')

def call_model(messages):
    url=api_base + '/chat/completions'
    body=json.dumps({'model':model,'messages':messages,'temperature':0.2}, ensure_ascii=False).encode('utf-8')
    headers={'Content-Type':'application/json'}
    if api_key:
        headers['Authorization']='Bearer '+api_key
    req=urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data=json.loads(resp.read().decode('utf-8'))
    return data['choices'][0]['message']['content']

def parse_file_blocks(text):
    blocks=[]
    pattern=re.compile(r'<<<FILE:(.*?)>>>\n(.*?)<<<END>>>', re.S)
    for path, content in pattern.findall(text):
        p=(project_dir/path.strip()).resolve()
        try:
            p.relative_to(project_dir)
        except ValueError:
            print(f'[跳过] 非法路径: {path}')
            continue
        blocks.append((p, content.rstrip('\n')+'\n'))
    return blocks

def apply_block(path, content):
    rel=path.relative_to(project_dir)
    parent=path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak=path.with_suffix(path.suffix+'.bak')
        shutil.copy2(path, bak)
        print(f'[备份] {rel} -> {rel}.bak')
    path.write_text(content, encoding='utf-8')
    print(f'[已修改] {rel}')

def show_help():
    print('命令:')
    print('  直接输入需求  -> 让大模型分析/修改工程')
    print('  files          -> 查看当前工程文件')
    print('  history        -> 查看本轮上下文消息数')
    print('  quit / exit    -> 退出')

def main():
    history=load_history()
    print('工程代码修改对话已启动。输入需求即可；退出输入 quit / exit。')
    show_help()
    messages=[]
    messages.append({'role':'system','content':system_prompt()})
    messages.extend(history)
    if not history:
        print('当前已是全新会话。')
    else:
        print(f'已恢复历史上下文，{len(history)} 条消息。')
    while True:
        try:
            user=input('\n[op_fix] > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出。')
            break
        if not user:
            continue
        if user.lower() in ('quit','exit'):
            print('退出。')
            break
        if user.lower() == 'files':
            for rel in sorted(str(p.relative_to(project_dir)) for p,_ in text_files(project_dir)):
                print(rel)
            continue
        if user.lower() == 'history':
            print('当前上下文消息数:', len(messages)-1)
            continue
        if user.lower() == 'help':
            show_help()
            continue
        messages.append({'role':'user','content':user})
        append_history({'role':'user','content':user})
        print('\n[AI] 正在思考...')
        try:
            content=call_model(messages)
        except Exception as e:
            print(f'[调用失败] {e}')
            continue
        print('\n[AI]\n'+content)
        messages.append({'role':'assistant','content':content})
        append_history({'role':'assistant','content':content})
        blocks=parse_file_blocks(content)
        if not blocks:
            continue
        print('\n检测到文件修改建议：')
        for i,(p,c) in enumerate(blocks,1):
            print(f'  {i}) {p.relative_to(project_dir)}  ({len(c)} bytes)')
        while True:
            try:
                ans=input('是否应用这些修改？选择编号(如 1,2 / all / n): ').strip()
            except (EOFError, KeyboardInterrupt):
                ans='n'
            if ans.lower() in ('n','no','q'):
                print('未应用修改。')
                break
            if ans.lower() in ('all','a','y'):
                for p,c in blocks:
                    apply_block(p,c)
                print('已应用全部修改。')
                break
            nums=[]
            for part in re.split(r'[,\s]+', ans):
                if part.isdigit():
                    idx=int(part)
                    if 1 <= idx <= len(blocks):
                        nums.append(idx)
            if not nums:
                print('输入无效，请重新输入。')
                continue
            for idx in nums:
                p,c=blocks[idx-1]
                apply_block(p,c)
            print('已应用所选修改。')
            break

main()
PY
OP_FIX_PROJECT="$PROJECT_DIR" \
OP_FIX_API_BASE="$API_BASE" \
OP_FIX_API_KEY="$API_KEY" \
OP_FIX_MODEL="$MODEL" \
OP_FIX_TIMEOUT="$TIMEOUT" \
python3 "$OP_FIX_PY"
