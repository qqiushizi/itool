#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""itool-server: 把 itool 菜单树以 HTTP 服务形式暴露，供远程 `curl | bash` 客户端使用。

服务端只读: 提供菜单结构、打包下载脚本目录、托管客户端脚本。
run.sh 始终在发起请求的本地机器上执行, 服务端绝不代跑。

用法:
    ITOOL_PASSWORD='your-password' python3 itool-server.py [ROOT_DIR] [PORT] [HOST]
    ITOOL_PASSWORD_FILE=/path/to/password python3 itool-server.py [ROOT_DIR] [PORT] [HOST]
    ITOOL_TOKEN='temporary-token' python3 itool-server.py [ROOT_DIR] [PORT] [HOST]

密码文件优先于 ITOOL_PASSWORD；服务端拒绝在未配置密码时启动。
HTTP 不加密密码，请仅用于受信网络；跨不可信网络时应置于 HTTPS 反向代理之后。
默认 ROOT_DIR=脚本所在目录, PORT=5170, HOST=0.0.0.0

远程使用 (在任意机器 B 上):
    curl -s http://<server-A>:5170/menu | bash
"""
import os
import re
import sys
import io
import tarfile
import socket
import hmac
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 5170 
HOST = "0.0.0.0"
CLIENT_FILE = os.path.join(ROOT_DIR, "menu")
PASSWORD = ""
STARTUP_TOKEN = ""
SESSION_TTL = 8 * 60 * 60
SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def load_password():
    """从环境变量或文件读取服务端密码，密码文件优先。"""
    password_file = os.environ.get("ITOOL_PASSWORD_FILE", "")
    if password_file:
        try:
            with open(password_file, "r", encoding="utf-8") as f:
                return f.read().rstrip("\r\n")
        except OSError as e:
            raise RuntimeError("无法读取 ITOOL_PASSWORD_FILE: %s" % e)
    return os.environ.get("ITOOL_PASSWORD", "")


def create_session():
    token = secrets.token_urlsafe(32)
    now = time.time()
    with SESSIONS_LOCK:
        expired = [key for key, deadline in SESSIONS.items() if deadline <= now]
        for key in expired:
            del SESSIONS[key]
        SESSIONS[token] = now + SESSION_TTL
    return token


def valid_session(token):
    if not token:
        return False
    if STARTUP_TOKEN and hmac.compare_digest(token, STARTUP_TOKEN):
        return True
    now = time.time()
    with SESSIONS_LOCK:
        deadline = SESSIONS.get(token, 0)
        if deadline <= now:
            SESSIONS.pop(token, None)
            return False
        return True


def safe_path(rel):
    """把相对路径解析到 ROOT_DIR 内的绝对路径, 拒绝目录穿越。"""
    rel = unquote(rel or "")
    root = os.path.abspath(ROOT_DIR)
    target = os.path.abspath(os.path.join(root, rel))
    if target != root and not target.startswith(root + os.sep):
        return None
    if not os.path.isdir(target):
        return None
    return target


def _num(field):
    """复刻 `sort -n`: 取前导数字, 无数字视为 0。"""
    m = re.match(r"\d+", field)
    return int(m.group()) if m else 0


def get_prefix(name):
    """复刻 itool.sh get_prefix: 第一个 '.' 前的部分, 转小写。"""
    return name.split(".", 1)[0].lower()


def list_menu(target):
    """复刻 itool.sh get_folders + `sort -t. -k1,1n`。"""
    has_run = os.path.isfile(os.path.join(target, "run.sh"))
    folders = [
        n for n in os.listdir(target)
        if not n.startswith(".") and os.path.isdir(os.path.join(target, n))
    ]
    folders.sort(key=lambda n: (_num(n.split(".", 1)[0]), n))
    return has_run, folders


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class Handler(BaseHTTPRequestHandler):
    server_version = "itool-server/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_bytes(self, data, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, ctype="text/plain; charset=utf-8", code=200, extra=None):
        self._send_bytes(text.encode("utf-8"), ctype, code, extra)

    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        path = q.get("path", [""])[0]
        route = parsed.path

        if route == "/menu":
            self._handle_client()
        elif route.startswith("/api/") and not self._authenticated():
            self._send_text(
                "Unauthorized\n",
                code=401,
                extra={"WWW-Authenticate": 'Bearer realm="itool"'},
            )
        elif route == "/api/menu":
            self._handle_menu(path)
        elif route == "/api/pack":
            self._handle_pack(path)
        elif route == "/api/cat":
            self._handle_cat(path)
        elif route in ("/", ""):
            self._handle_index()
        else:
            self._send_text("Not Found\n", code=404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/login":
            self._send_text("Not Found\n", code=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 1024:
            self._send_text("Invalid credential\n", code=400)
            return
        supplied = self.rfile.read(length).decode("utf-8", errors="replace")
        password_matches = hmac.compare_digest(supplied, PASSWORD)
        token_matches = bool(STARTUP_TOKEN) and hmac.compare_digest(
            supplied, STARTUP_TOKEN
        )
        if not (password_matches or token_matches):
            self._send_text(
                "密码或临时 token 错误\n",
                code=401,
                extra={"Cache-Control": "no-store"},
            )
            return
        self._send_text(create_session() + "\n", extra={"Cache-Control": "no-store"})

    def _authenticated(self):
        authorization = self.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        return scheme.lower() == "bearer" and valid_session(token.strip())

    def _handle_client(self):
        try:
            with open(CLIENT_FILE, "r", encoding="utf-8") as f:
                tpl = f.read()
        except OSError:
            self._send_text("menu 模板缺失\n", code=500)
            return
        host = self.headers.get("Host") or ("%s:%d" % (HOST, PORT))
        proto = self.headers.get("X-Forwarded-Proto", "http").split(",", 1)[0].strip()
        if proto not in ("http", "https"):
            proto = "http"
        base = proto + "://" + host
        out = tpl.replace("__SERVER_URL__", base)
        self._send_bytes(
            out.encode("utf-8"),
            "text/x-shellscript; charset=utf-8",
        )

    def _handle_menu(self, path):
        target = safe_path(path)
        if not target:
            self._send_text("invalid path\n", code=400)
            return
        has_run, folders = list_menu(target)
        lines = ["HAS_RUN\t%d" % (1 if has_run else 0)]
        for n in folders:
            lines.append("FOLDER\t%s\t%s" % (n, get_prefix(n)))
        self._send_text("\n".join(lines) + "\n")

    def _handle_pack(self, path):
        target = safe_path(path)
        if not target:
            self._send_text("invalid path\n", code=400)
            return
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for item in os.listdir(target):
                tar.add(os.path.join(target, item), arcname=item)
        data = buf.getvalue()
        self._send_bytes(
            data,
            "application/gzip",
            extra={"Content-Disposition": 'attachment; filename="itool-pack.tar.gz"'},
        )

    def _handle_cat(self, path):
        target = safe_path(path)
        if not target:
            self._send_text("invalid path\n", code=400)
            return
        run = os.path.join(target, "run.sh")
        if not os.path.isfile(run):
            self._send_text("no run.sh\n", code=404)
            return
        try:
            with open(run, "r", encoding="utf-8", errors="replace") as f:
                self._send_text(f.read())
        except OSError as e:
            self._send_text("read error: %s\n" % e, code=500)

    def _handle_index(self):
        ip = local_ip()
        msg = (
            "itool-server 运行中\n"
            "====================\n\n"
            "在远程机器上执行以下命令即可获得菜单 (客户端零安装):\n\n"
            "    curl -s http://%s:%d/menu | bash\n\n"
            "API:\n"
            "  /menu            客户端脚本 (SERVER_URL 自动注入)\n"
            "  POST /api/login       密码登录并获取临时会话\n"
            "  /api/menu?path=...    菜单结构 (制表符分隔, 免 jq)\n"
            "  /api/pack?path=...    打包下载 run.sh 所在目录 (tar.gz)\n"
            "  /api/cat?path=...     预览 run.sh 内容\n"
        ) % (ip, PORT)
        self._send_text(msg)


def main():
    global ROOT_DIR, PORT, HOST, PASSWORD, STARTUP_TOKEN, SESSION_TTL
    args = sys.argv[1:]
    if len(args) >= 1:
        ROOT_DIR = os.path.abspath(args[0])
    if len(args) >= 2:
        PORT = int(args[1])
    if len(args) >= 3:
        HOST = args[2]
    try:
        PASSWORD = load_password()
    except RuntimeError as e:
        print("错误: %s" % e, file=sys.stderr)
        sys.exit(1)
    if not PASSWORD:
        print(
            "错误: 未配置服务端密码，请设置 ITOOL_PASSWORD 或 ITOOL_PASSWORD_FILE。",
            file=sys.stderr,
        )
        sys.exit(1)
    STARTUP_TOKEN = os.environ.get("ITOOL_TOKEN", "")
    if STARTUP_TOKEN and not re.fullmatch(r"[0-9]{6}", STARTUP_TOKEN):
        print("错误: ITOOL_TOKEN 必须是 6 位数字。", file=sys.stderr)
        sys.exit(1)
    try:
        SESSION_TTL = int(os.environ.get("ITOOL_SESSION_TTL", str(SESSION_TTL)))
        if SESSION_TTL <= 0:
            raise ValueError
    except ValueError:
        print("错误: ITOOL_SESSION_TTL 必须是正整数秒数。", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(ROOT_DIR):
        print("ROOT_DIR 不存在: %s" % ROOT_DIR)
        sys.exit(1)
    if not os.path.isfile(CLIENT_FILE):
        print("警告: 未找到 menu, /menu 将返回 500")

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    ip = local_ip()
    print("itool-server 启动")
    print("  监听: %s:%d" % (HOST, PORT))
    print("  ROOT: %s" % ROOT_DIR)
    token_status = "可用" if STARTUP_TOKEN else "未配置"
    print(
        "  认证: 已启用 (密码会话有效期 %d 秒，启动 token %s)"
        % (SESSION_TTL, token_status)
    )
    print("  远程使用 (在 B 机器上):")
    print("    curl -s http://%s:%d/menu | bash" % (ip, PORT))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n停止")
        srv.shutdown()


if __name__ == "__main__":
    main()
