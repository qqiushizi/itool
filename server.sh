#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SERVER_SCRIPT="$SCRIPT_DIR/itool-server.py"
TOKEN_FILE="$SCRIPT_DIR/token.txt"
PID_FILE="$SCRIPT_DIR/server.pid"
LOG_FILE="$SCRIPT_DIR/server.log"

is_server_process() {
    local pid="$1" arg
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
    while IFS= read -r -d '' arg; do
        [[ "$arg" == "$SERVER_SCRIPT" ]] && return 0
    done < "/proc/$pid/cmdline"
    return 1
}

stop_existing_servers() {
    local process_dir pid attempt still_running
    local -a server_pids=()

    for process_dir in /proc/[0-9]*; do
        pid="${process_dir##*/}"
        if is_server_process "$pid"; then
            server_pids+=("$pid")
        fi
    done

    if [[ ${#server_pids[@]} -eq 0 ]]; then
        rm -f "$PID_FILE"
        return 0
    fi

    echo "正在停止原有服务进程: ${server_pids[*]}"
    for pid in "${server_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    for attempt in {1..50}; do
        still_running=0
        for pid in "${server_pids[@]}"; do
            if is_server_process "$pid"; then
                still_running=1
                break
            fi
        done
        [[ $still_running -eq 0 ]] && break
        sleep 0.1
    done

    for pid in "${server_pids[@]}"; do
        if is_server_process "$pid"; then
            echo "原有进程未及时退出，强制停止: $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    rm -f "$PID_FILE"
}

# 每次启动生成新的临时访问 token，仅允许当前用户读取。
ITOOL_TOKEN=$(python3 -c 'import secrets; print(f"{secrets.randbelow(1000000):06d}")') || {
    echo "无法生成临时 token。" >&2
    exit 1
}
if [[ ! "$ITOOL_TOKEN" =~ ^[0-9]{6}$ ]]; then
    echo "生成的临时 token 不是 6 位数字。" >&2
    exit 1
fi
umask 077
if ! printf '%s\n' "$ITOOL_TOKEN" > "$TOKEN_FILE"; then
    echo "无法写入临时 token: $TOKEN_FILE" >&2
    exit 1
fi
if ! chmod 600 "$TOKEN_FILE"; then
    echo "无法设置临时 token 文件权限: $TOKEN_FILE" >&2
    exit 1
fi
export ITOOL_TOKEN

if [[ -z "$ITOOL_PASSWORD" && -z "$ITOOL_PASSWORD_FILE" ]]; then
    if ! { : < /dev/tty; } 2>/dev/null; then
        echo "请通过 ITOOL_PASSWORD 或 ITOOL_PASSWORD_FILE 配置服务端密码。" >&2
        exit 1
    fi
    IFS= read -r -s -p "设置服务端密码: " ITOOL_PASSWORD < /dev/tty
    echo > /dev/tty
    if [[ -z "$ITOOL_PASSWORD" ]]; then
        echo "服务端密码不能为空。" >&2
        exit 1
    fi
    export ITOOL_PASSWORD
fi

stop_existing_servers

echo "临时密码（token）: $ITOOL_TOKEN"
echo "token 文件: $TOKEN_FILE"

PYTHONUNBUFFERED=1 nohup python3 "$SERVER_SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$PID_FILE"
chmod 600 "$PID_FILE"

sleep 0.5
if ! is_server_process "$SERVER_PID"; then
    rm -f "$PID_FILE"
    echo "服务启动失败，日志如下:" >&2
    tail -n 20 "$LOG_FILE" >&2
    exit 1
fi

echo "服务已启动，PID: $SERVER_PID"
echo "运行日志: $LOG_FILE"
