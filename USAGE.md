# itool 使用文档（照抄就能跑）

> 本文件是「怎么用」的具体操作手册；架构与目录说明见 [README.md](README.md)。

## 0. 前置条件

| 用途 | 需要 |
|---|---|
| 本地导航 | 任意 Linux / macOS，`bash` |
| 远程服务端 | Linux + `python3`（标准库即可） |
| 远程客户端 | `bash` + `curl` |
| 算子开发/安装类脚本 | 昇腾环境：CANN toolkit、docker、torch/torch_npu 等 |

---

## 1. 本地使用（文件夹导航器）

```bash
cd itool
bash itool.sh        # 直接运行
# 或
source itool.sh      # source 运行, 退出后自动 cd 到所选目录
```

进入后按 `d` 快速跳到 `d.ops_develop`，用 `↑↓` 移动、`Enter` 进入/执行、`Backspace` 返回、`*` 回根、`ESC` 退出。

真实交互长这样：

```
════════════════════════════════════════════════════
  📁 文件夹导航器
════════════════════════════════════════════════════
  📂 /
────────────────────────────────────────────────────
  ▶ a a.framework
    [b] b.performance
    [c] c.precision
    [d] d.ops_develop
    [e] e.environment
    ...
────────────────────────────────────────────────────
  ↑↓ 移动  Enter 确认  Bksp 返回  * 根目录  ESC 退出
```

叶子目录里选项 `0` 就是 `run.sh`，回车即执行。

---

## 2. 远程使用（服务器 A 托管，机器 B 零安装执行）

### 2.1 服务端（机器 A）

```bash
cd itool
export ITOOL_PASSWORD='MySecret123'   # 或 export ITOOL_PASSWORD_FILE=/path/to/pw
bash server.sh
```

输出：

```
临时密码（token）: 405298
token 文件: /data2/lxy/itool/token.txt
服务已启动，PID: 2151193
运行日志: /data2/lxy/itool/server.log
```

> `405298` 是本次启动的临时 token，可发给同事跳过密码登录。

### 2.2 客户端（机器 B，三种方式任选）

```bash
# 方式一: 密码登录(交互, 输 MySecret123)
curl -s http://<server-A>:5170/menu | bash

# 方式二: 用临时 token 跳过密码
curl -s http://<server-A>:5170/menu | ITOOL_TOKEN=405298 bash

# 方式三: 手动指定服务器地址
ITOOL_SERVER=http://<server-A>:5170 bash menu
```

进入后和本地一样：方向键选目录 → 回车执行 `run.sh`。**run.sh 是在机器 B 本地执行的**（服务端只下发脚本打包，不代跑）。

### 2.3 HTTP API（编程接入用）

```bash
# 登录拿会话 token
curl -s -d 'MySecret123' http://<server-A>:5170/api/login
# → 7cyM8gTe2nuBTtzhnOimwX9fFAgZn5kqSGC6fW9Nvq8

# 菜单结构
curl -s -H 'Authorization: Bearer <token>' 'http://<server-A>:5170/api/menu?path=d.ops_develop'
# → HAS_RUN	0
#   FOLDER	b.image_container	a
#   FOLDER	c.env_check	b
#   FOLDER	d.install_cann	c
#   ...

# 打包下载 + 预览
curl -s -H 'Authorization: Bearer <token>' -o pack.tar.gz 'http://<server-A>:5170/api/pack?path=e.environment/a.versions/a.all'
curl -s -H 'Authorization: Bearer <token>' 'http://<server-A>:5170/api/cat?path=e.environment/a.versions/a.all'
```

---

## 3. 算子开发工作流（d.ops_develop）

面向：在客户机器上开发昇腾算子。链路收敛为 6 项：

```
① a.llm_config       可选：大模型配置管理
② b.image_container  拉镜像 + 建容器
③ c.env_check        容器内环境检查（只读，只给建议）
④ d.install_cann     按需安装/补装 CANN toolkit（检查通过可跳过）
⑤ e.install_opencode 安装 OpenCode + CANNBot skills/agents
⑥ f.op_agent         打开 OpenCode 开发 → 生成测试 → 测试 → 报告
```

### 3.1 ② 镜像拉取 + 容器实例化

```bash
bash d.ops_develop/b.image_container/run.sh
```

进入容器：

```bash
docker exec -it asc_dev bash
cd /workspace/itool
```

### 3.2 ③ 容器内环境检查（只读，只给建议）

```bash
bash d.ops_develop/c.env_check/run.sh
```

### 3.3 ④ 按需安装/补装 CANN toolkit（容器内）

```bash
bash d.ops_develop/d.install_cann/a.cann-9.1.0/run.sh
```

### 3.4 ⑤ 安装 OpenCode + CANNBot skills/agents

```bash
# 安装 opencode 绿色版
bash d.ops_develop/e.install_opencode/a.install_opencode/run.sh

# 安装 CANNBot skills/agents
bash d.ops_develop/e.install_opencode/b.install_cannbot_skill/run.sh
```

`e.install_opencode/b.install_cannbot_skill` 默认优先走官方 CANNBot 安装助手；网络不可用时回退到绿色包内置 skills。

### 3.5 ⑥ 算子 Agent 开发/测试/报告

```bash
bash d.ops_develop/f.op_agent/run.sh
```

菜单包含：

1. 选择/新建算子工程
2. 打开 OpenCode 开发
3. 生成测试用例
4. 编译工程
5. 运行测试
6. 生成测试报告

生成物归档：

```text
d.ops_develop/workspace/op_build_<算子名>/
└── report/
    ├── build.log
    ├── test.log
    └── test_report.md
```

## 4. 常见问题

| 现象 | 处理 |
|---|---|
| `curl \| bash` 报"需要交互式终端" | 客户端交互走 `/dev/tty`，请在真实终端运行 |
| CANN 下载 403 | 先 `CHECK_ONLY=1` 探测；不可达则用 `CANN_BASE_URL` 换内网源 |
| `itool.sh` 提示 `./isetenv.sh` 不存在 | 已加守卫不阻塞；如需自动加载环境，把 `e.environment/e.setenvs/setenvs.sh` 拷成根目录 `isetenv.sh` |
| 起服务没密码 | 先 `export ITOOL_PASSWORD=xxx`，否则首次会交互设置 |
| 下载 GLM-5.1 | 先 `export OPENMIND_HUB_TOKEN=xxx` |

---

## 5. 目录速查

| 目录 | 用途 |
|---|---|
| `a.framework` | 训练/推理框架部署 |
| `c.precision` | 精度对齐 |
| `d.ops_develop` | 算子开发工作流 |
| `e.environment` | 环境检查/配置 |
| `f.installation` | 各类安装 |
| `g.collectlogs` | 日志采集 |
| `h.auto-test` | 自动化测试 |
| `p.practise` | AI 学习实验(141 个) |
| `w.windows` | WSL 安装 |
