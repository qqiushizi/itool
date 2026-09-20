#!/bin/bash
# ============================================================
# ⑥ 算子 Agent 开发：OpenCode 开发 -> 生成测试 -> 测试 -> 测试报告
# 用法:
#   ./run.sh
#   MODE=auto ./run.sh [算子名]
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo "$PWD")
source "$SCRIPT_DIR/../llm_profile.sh"
resolve_ops_root "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; RESET='\033[0m'

ask() {
    local p="$1" d="$2"
    printf "  %s [%s]: " "$p" "$d"
    IFS= read -r REPLY || REPLY=""
    [ -n "$REPLY" ] || REPLY="$d"
}
confirm() {
    local ans
    printf "  %s [y/N]: " "$1"
    IFS= read -r ans || ans=""
    case "$ans" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

# ---------- 查找 opencode ----------
find_opencode() {
    OPCODE_CMD="${OPCODE_CMD:-}"
    OPCODE_ROOT="${OPCODE_ROOT:-}"
    if [ -n "$OPCODE_CMD" ]; then
        return 0
    fi
    if command -v iopencode >/dev/null 2>&1; then
        OPCODE_CMD="iopencode"
    elif command -v opencode >/dev/null 2>&1; then
        OPCODE_CMD="opencode"
    elif [ -x "$SCRIPT_DIR/../e.install_opencode/a.install_opencode/opencode/start.sh" ]; then
        OPCODE_CMD="$SCRIPT_DIR/../e.install_opencode/a.install_opencode/opencode/start.sh"
        OPCODE_ROOT="$(cd "$(dirname "$OPCODE_CMD")/.." && pwd)"
    fi
    [ -n "$OPCODE_CMD" ] || return 1
}

# ---------- 工程选择 ----------
list_projects() {
    PROJECTS=()
    local d
    for d in "$OPS_WORKSPACE"/op_build_*; do
        [ -d "$d" ] || continue
        PROJECTS+=("$d")
    done
}

select_or_create_project() {
    list_projects
    if [ "${1:-}" = "new" ]; then
        create_new_project
        return $?
    fi
    if [ "${#PROJECTS[@]}" -eq 0 ]; then
        echo -e "  ${YELLOW}当前没有算子工程，将新建。${RESET}"
        create_new_project
        return $?
    fi
    echo -e "  ${CYAN}请选择或新建算子工程:${RESET}"
    local i
    for ((i=0; i<${#PROJECTS[@]}; i++)); do
        printf '    %3d) %s\n' "$((i+1))" "$(basename "${PROJECTS[$i]}")"
    done
    echo -e "    ${GREEN}N${RESET}) 新建工程"
    while true; do
        ask "选择" "1"
        case "$(printf '%s' "$REPLY" | tr '[:upper:]' '[:lower:]')" in
            n) create_new_project; return $? ;;
        esac
        if [ "$REPLY" -ge 1 ] 2>/dev/null && [ "$REPLY" -le "${#PROJECTS[@]}" ] 2>/dev/null; then
            PROJECT_DIR="${PROJECTS[$((REPLY-1))]}"
            break
        fi
        echo -e "  ${YELLOW}编号无效。${RESET}"
    done
}

create_new_project() {
    local name
    if [ -n "${1:-}" ]; then
        name="$1"
    else
        while true; do
            ask "算子名称(如 AddCustom/MatMulCustom)" ""
            name="$REPLY"
            [ -n "$name" ] && break
        done
    fi
    PROJECT_DIR="$OPS_WORKSPACE/op_build_$name"
    if [ -d "$PROJECT_DIR" ] && [ -n "$(ls -A "$PROJECT_DIR" 2>/dev/null)" ]; then
        echo -e "  ${GREEN}工程已存在，使用: $PROJECT_DIR${RESET}"
    else
        mkdir -p "$PROJECT_DIR"
        echo -e "  ${GREEN}已创建工程目录: $PROJECT_DIR${RESET}"
    fi
    write_agents_md
}

write_agents_md() {
    local target="$PROJECT_DIR/AGENTS.md"
    local desc
    if [ -n "${OP_DESC:-}" ]; then
        desc="$OP_DESC"
    else
        ask "一句话描述算子需求" "开发一个可编译运行的 AscendC/C++ 算子"
        desc="$REPLY"
    fi
    local arch
    ask "目标芯片" "Ascend910B"
    arch="$REPLY"
    cat > "$target" <<MD
---
description: AscendC/C++ 算子开发任务
mode: primary
---

# 算子开发任务

- 算子目录: $PROJECT_DIR
- 目标芯片: $arch
- 需求描述: $desc

## 要求
1. 在本目录内完成 AscendC/C++ 算子工程开发。
2. 必须保证工程可编译: cd $PROJECT_DIR && bash build.sh。
3. 生成必要的测试用例、测试脚本和输入数据。
4. 修改完成后给出实际执行命令和关键输出。
5. 不修改本工程目录以外的文件。
MD
    echo -e "  ${GREEN}已写入任务说明: $target${RESET}"
}

# ---------- 打开开发 ----------
do_develop() {
    find_opencode || {
        echo -e "${RED}未找到 opencode，请先执行安装功能 e.install_opencode。${RESET}" >&2
        return 1
    }
    if [ -z "${PROJECT_DIR:-}" ]; then
        select_or_create_project
    fi
    echo -e "  ${CYAN}启动 OpenCode 开发(项目: $PROJECT_DIR)${RESET}"
    cd "$PROJECT_DIR"
    if [ -n "$OPCODE_ROOT" ]; then
        export XDG_CONFIG_HOME="$OPCODE_ROOT/config"
    fi
    "$OPCODE_CMD"
}

# ---------- 非交互 agent 调用 ----------
run_agent_prompt() {
    local prompt="$1"
    find_opencode || return 1
    cd "$PROJECT_DIR"
    if [ -n "$OPCODE_ROOT" ]; then
        export XDG_CONFIG_HOME="$OPCODE_ROOT/config"
    fi
    "$OPCODE_CMD" run "$prompt"
}

# ---------- 生成测试用例 ----------
do_gen_tests() {
    [ -z "${PROJECT_DIR:-}" ] && select_or_create_project
    run_agent_prompt "请根据当前算子工程的 op.json / 现有代码，生成完整的测试用例、测试脚本和测试输入数据；测试覆盖前向一次运行即可。"
    echo -e "  ${GREEN}测试用例生成指令已执行。${RESET}"
}

# ---------- 编译 ----------
do_build() {
    [ -z "${PROJECT_DIR:-}" ] && select_or_create_project
    local log="$PROJECT_DIR/report/build.log"
    mkdir -p "$(dirname "$log")"
    echo "==> 编译项目: $PROJECT_DIR"
    ( cd "$PROJECT_DIR" && bash build.sh ) > "$log" 2>&1
    local rc=$?
    echo "    build.sh 退出码: $rc"
    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}编译成功。${RESET}"
    else
        echo -e "  ${RED}编译失败，日志: $log${RESET}"
    fi
    BUILD_RC=$rc
}

# ---------- 运行测试 ----------
do_test() {
    [ -z "${PROJECT_DIR:-}" ] && select_or_create_project
    local log="$PROJECT_DIR/report/test.log"
    mkdir -p "$(dirname "$log")"
    local test_script
    test_script=$(find "$PROJECT_DIR" -maxdepth 3 -type f \( -name 'run_test.sh' -o -name 'test.sh' -o -name 'run_ut.sh' \) | head -1 || true)
    if [ -z "$test_script" ]; then
        echo -e "  ${YELLOW}未找到测试脚本，请先执行“生成测试用例”。${RESET}" >&2
        TEST_RC=-1
        : > "$log"
        return 0
    fi
    echo "==> 执行测试脚本: $test_script"
    ( cd "$PROJECT_DIR" && bash "$test_script" ) > "$log" 2>&1
    local rc=$?
    echo "    测试脚本退出码: $rc"
    if [ $rc -eq 0 ]; then
        echo -e "  ${GREEN}测试通过。${RESET}"
    else
        echo -e "  ${RED}测试失败，日志: $log${RESET}"
    fi
    TEST_RC=$rc
}

# ---------- 测试报告 ----------
do_report() {
    [ -z "${PROJECT_DIR:-}" ] && select_or_create_project
    local report="$PROJECT_DIR/report/test_report.md"
    mkdir -p "$(dirname "$report")"
    local build_log="$PROJECT_DIR/report/build.log"
    local test_log="$PROJECT_DIR/report/test.log"
    local build_status="未执行"
    local test_status="未执行"
    if [ -f "$build_log" ]; then
        if grep -q "build.sh 退出码: 0" "$build_log" 2>/dev/null || [ "${BUILD_RC:-}" = "0" ]; then
            build_status="通过"
        else
            build_status="失败"
        fi
    fi
    if [ -f "$test_log" ]; then
        if [ "${TEST_RC:-}" = "0" ]; then
            test_status="通过"
        elif [ "${TEST_RC:-}" = "-1" ]; then
            test_status="未生成测试脚本"
        elif [ -s "$test_log" ]; then
            test_status="失败"
        fi
    fi
    cat > "$report" <<MD
# 算子测试报告

- 算子工程: $PROJECT_DIR
- 生成时间: $(date '+%Y-%m-%d %H:%M:%S')
- 编译结果: $build_status
- 测试结果: $test_status

## 编译日志

\`\`\`text
$(tail -n 60 "$build_log" 2>/dev/null || true)
\`\`\`

## 测试日志

\`\`\`text
$(tail -n 80 "$test_log" 2>/dev/null || true)
\`\`\`
MD
    echo -e "  ${GREEN}测试报告已生成: $report${RESET}"
}

# ---------- 主流程 ----------
BUILD_RC=""
TEST_RC=""

if [ "${MODE:-menu}" = "auto" ]; then
    name="${1:-}"
    [ -n "$name" ] || { echo -e "${RED}MODE=auto 需要传入算子名。${RESET}" >&2; exit 1; }
    create_new_project "$name"
    do_build
    do_gen_tests
    do_build
    do_test
    do_report
    exit $?
fi

while true; do
    echo ""
    echo -e "  ${WHITE}===== 算子 Agent 开发 =====${RESET}"
    if [ -n "${PROJECT_DIR:-}" ]; then
        echo -e "  当前工程: ${CYAN}$PROJECT_DIR${RESET}"
    fi
    echo ""
    echo -e "    1) 选择/新建算子工程"
    echo -e "    2) 打开 OpenCode 开发"
    echo -e "    3) 生成测试用例"
    echo -e "    4) 编译工程"
    echo -e "    5) 运行测试"
    echo -e "    6) 生成测试报告"
    echo -e "    Q) 退出"
    ask "请选择" "1"
    case "$REPLY" in
        1) select_or_create_project ;;
        2) do_develop ;;
        3) do_gen_tests ;;
        4) do_build ;;
        5) do_test ;;
        6) do_report ;;
        q|Q) echo -e "  ${GREEN}退出。${RESET}"; exit 0 ;;
        *) echo -e "  ${YELLOW}无效选择。${RESET}" ;;
    esac
done
