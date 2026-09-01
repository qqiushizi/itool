#!/bin/bash
#
# itool.sh - 纯终端交互式文件夹导航
#
if [[ -f ./isetenv.sh ]]; then
    source ./isetenv.sh
fi

ROOT_DIR="${1:-${FOLDER_NAV_ROOT:-.}}"
MAX_DEPTH=20

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
RESET='\033[0m'

# 全局变量：保存当前导航到的文件夹
GLOBAL_CURRENT_DIR=""

# 检测是否被 source（return 在子进程中会失败）
_sourced=0
(return 0 2>/dev/null) && _sourced=1

# 保存/恢复终端
save_tty() { old_tty=$(stty -g); }
restore_tty() {
    stty "$old_tty" 2>/dev/null
    if [[ -n "$GLOBAL_CURRENT_DIR" && -d "$GLOBAL_CURRENT_DIR" ]]; then
        if [[ $_sourced -eq 1 ]]; then
            cd "$GLOBAL_CURRENT_DIR"
        else
            echo -e "  ${CYAN}📍 跳转到: ${GLOBAL_CURRENT_DIR}${RESET}"
            echo -e "  ${DIM}提示: 使用 ${YELLOW}source itool.sh${RESET}${DIM} 可自动留在该目录${RESET}"
        fi
    fi
    clear
}
trap 'restore_tty' EXIT INT TERM

# 获取排序后的文件夹
get_folders() {
    local dir="$1"
    while IFS= read -r f; do
        [[ -z "$f" || "$f" == .* || ! -d "$dir/$f" ]] && continue
        echo "$f"
    done < <(ls -1 "$dir" 2>/dev/null) | sort -t'.' -k1,1n
}

# 获取文件夹前缀
get_prefix() {
    echo "${1%%.*}" | tr '[:upper:]' '[:lower:]'
}

navigate() {
    local current="$ROOT_DIR"
    GLOBAL_CURRENT_DIR="$current"
    local stack=("$ROOT_DIR")
    local depth=0
    local selected=0
    save_tty
    
    while true; do
        # 收集选项
        local options=() prefixes=()
        local has_runscript=false
        [[ -f "$current/run.sh" ]] && has_runscript=true
        
        if $has_runscript; then
            options+=("run.sh")
            prefixes+=("0")
        fi
        
        while IFS= read -r folder; do
            options+=("$folder")
            prefixes+=("$(get_prefix "$folder")")
        done < <(get_folders "$current")
        
        local total=${#options[@]}
        [[ $selected -ge $total ]] && selected=0
        [[ $selected -lt 0 ]] && selected=0
        
        # ========== 渲染 ==========
        clear
        echo ""
        echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
        echo -e "  ${WHITE}  📁 文件夹导航器${RESET}"
        echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
        
        local path_display="${current#$ROOT_DIR}"
        [[ "$path_display" == "$current" || -z "$path_display" ]] && path_display="/"
        echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
        echo -e "  📂 ${path_display}"
        echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
        
        for ((i=0; i<${#options[@]}; i++)); do
            if [[ $i -eq $selected ]]; then
                echo -e "  ${GREEN}▶${RESET} ${prefixes[$i]} ${WHITE}${options[$i]}${RESET}"
            else
                echo -e "    ${DIM}[${prefixes[$i]}]${RESET} ${options[$i]}"
            fi
        done
        
        for ((i=${#options[@]}; i<8; i++)); do echo ""; done
        
        echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
        echo -e "  ${YELLOW}↑↓${RESET} 移动  ${YELLOW}Enter${RESET} 确认  ${YELLOW}Bksp${RESET} 返回  ${YELLOW}*${RESET} 根目录  ${YELLOW}ESC${RESET} 退出"
        echo ""
        
        # ========== 读取按键 ==========
        stty -icanon -echo min 1 time 0 2>/dev/null
        local key=""
        IFS= read -r -n 1 key
        
        # 如果是ESC，读取后续 (bash3.2 不支持 -t 0.1, 用 VMIN=0/VTIME=1 约100ms探测)
        if [[ "$key" == $'\e' ]]; then
            local sec=""
            stty min 0 time 1 2>/dev/null
            sec=$(dd bs=1 count=1 2>/dev/null)
            if [[ -n "$sec" ]]; then
                key="$key$sec"
                if [[ "$sec" == "[" || "$sec" == "O" ]]; then
                    sec=$(dd bs=1 count=1 2>/dev/null)
                    key="$key$sec"
                fi
            fi
            stty min 1 time 0 2>/dev/null
        fi
        stty icanon echo 2>/dev/null
        
        # ========== 处理按键 ==========
        case "$key" in
            $'\e[A')  # 上
                ((selected--))
                [[ $selected -lt 0 ]] && selected=$((total-1))
                continue
                ;;
            $'\e[B')  # 下
                ((selected++))
                [[ $selected -ge $total ]] && selected=0
                continue
                ;;
            $'\b'|$'\177')  # Backspace 返回上一层
                if [[ ${#stack[@]} -gt 1 ]]; then
                    unset "stack[$(( ${#stack[@]} - 1 ))]"
                    current="${stack[$(( ${#stack[@]} - 1 ))]}"
                    GLOBAL_CURRENT_DIR="$current"
                    ((depth--))
                    selected=0
                fi
                continue
                ;;
            $'\e')  # 单独ESC键 = 退出
                clear
                echo -e "  ${CYAN}👋 再见!${RESET}"
                echo
                [[ $_sourced -eq 1 && -n "$GLOBAL_CURRENT_DIR" && -d "$GLOBAL_CURRENT_DIR" ]] && cd "$GLOBAL_CURRENT_DIR"
                if [[ $_sourced -eq 1 ]]; then
                    return 0 2>/dev/null
                else
                    exit 0
                fi
                ;;
            ""|$'\n'|$'\r')  # Enter / 空
                ;;
            \*)
                current="$ROOT_DIR"
                GLOBAL_CURRENT_DIR="$current"
                stack=("$ROOT_DIR")
                depth=0
                selected=0
                continue
                ;;
            *)
                # 数字/字母快速跳转
                local k
                k=$(printf '%s' "$key" | tr '[:upper:]' '[:lower:]')
                [[ ! "$k" =~ ^[0-9a-z]$ ]] && continue
                
                local found=-1
                for ((i=0; i<${#prefixes[@]}; i++)); do
                    [[ "${prefixes[$i]}" == "$k" ]] && found=$i
                done
                [[ $found -lt 0 ]] && continue
                selected=$found
                ;;
        esac
        
        # ========== 执行选中项 ==========
        local choice="${options[$selected]}"
        
        if [[ "$choice" == "run.sh" ]]; then
            clear
            echo ""
            echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
            echo -e "  ${GREEN}⚙ 执行 run.sh...  ${DIM}(按 Ctrl+C 中断)${RESET}"
            echo -e "  ${CYAN}════════════════════════════════════════════════════${RESET}"
            echo ""

            stty icanon echo 2>/dev/null
            local interrupted=0
            local ec=0

            # 终端保持 icanon+echo，使 run.sh 内部的交互输入（read）能正常回显。
            # 中断改用 Ctrl+C：父进程捕获 SIGINT 置中断标志，子进程重置为默认处置从而被终止。
            trap 'interrupted=1' INT
            ( trap - INT; cd "$current" && bash run.sh )
            ec=$?
            [[ $ec -eq 130 ]] && interrupted=1
            trap 'restore_tty' INT
            stty icanon echo 2>/dev/null

            echo ""
            echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
            if [[ $interrupted -eq 1 ]]; then
                echo -e "  ${YELLOW}⏹ 已中断${RESET} (退出码: $ec)"
            else
                echo -e "  ${GREEN}✓ 完成${RESET} (退出码: $ec)"
            fi
            echo -e "  ${CYAN}────────────────────────────────────────────────────${RESET}"
            echo -ne "  ${DIM}按任意键继续...${RESET}"
            stty -icanon -echo 2>/dev/null
            IFS= read -r -n 1
            stty icanon echo 2>/dev/null

            if [[ $interrupted -eq 1 ]]; then
                if [[ $_sourced -eq 1 ]]; then
                    return 0 2>/dev/null
                else
                    exit 0
                fi
            fi
            
        elif [[ -d "$current/$choice" ]]; then
            current="$current/$choice"
            GLOBAL_CURRENT_DIR="$current"
            stack+=("$current")
            ((depth++))
            selected=0

            if [[ $depth -ge $MAX_DEPTH ]]; then
                echo -e "  ${RED}⚠ 达到最大深度!${RESET}"
                sleep 1
                unset "stack[$(( ${#stack[@]} - 1 ))]"
                current="${stack[$(( ${#stack[@]} - 1 ))]}"
                GLOBAL_CURRENT_DIR="$current"
                ((depth--))
            fi
        fi
    done
}

main() {
    if [[ ! -d "$ROOT_DIR" ]]; then
        echo -e "${RED}错误: 目录 '$ROOT_DIR' 不存在${RESET}"
        exit 1
    fi
    
    navigate
}

main "$@"
