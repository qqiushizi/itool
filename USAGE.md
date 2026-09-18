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

面向：在客户机器上开发算子 / 基于算子源码改造。完整链路共 8 步，第 8 步为可选安装 OpenCode/CANNBot agent。

```
① a.llm_config      创建/选择大模型配置
② b.image_container 拉镜像 + 建容器
③ c.env_check       容器内环境检查（只读，只给建议）
④ d.install_cann    按需安装/补装 CANN toolkit（检查通过可跳过）
⑤ e.op_design       选择模型配置/手动填写 → op.json + op_spec.md
⑥ f.op_build        用 msopgen 生成 AscendC 算子工程
⑦ g.op_fix          大模型辅助修改算子工程代码
⑧ h.op_agent        安装 OpenCode + CANNBot skills/agents
```

### 3.1 ① 大模型配置管理

```bash
bash d.ops_develop/a.llm_config/run.sh
```

用于：

- 新建配置
- 修改配置
- 删除配置
- 测试配置
- 设置当前使用的配置

配置保存在：

```text
d.ops_develop/workspace/llm_configs/<配置名>.json
```

后续 `e.op_design` 和 `g.op_fix` 只选择配置，不再要求用户重新输入 API Base / Key / Model。

### 3.2 ② 镜像拉取 + 容器实例化

```bash
bash d.ops_develop/b.image_container/run.sh
```

脚本会询问镜像地址和容器名称（默认 `asc_dev`）。也支持非交互：

```bash
IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 bash d.ops_develop/b.image_container/run.sh

bash d.ops_develop/b.image_container/run.sh quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 asc_dev
```

自动流程：

1. `docker image inspect` 检查本机是否已有该镜像
2. 不存在则自动 `docker pull`
3. 获取镜像稳定 ID
4. 自动检测当前机器存在的 `/dev/davinci*`、驱动目录、`dcmi`、`npu-smi`
5. 创建工作目录并生成 `start_container.sh`
6. 立即启动容器

默认宿主机工作目录：

```text
~/ascend_ops_workspace
```

挂载关系：

```text
宿主机 ~/ascend_ops_workspace  →  容器 /workspace
```

可通过 `WORK_DIR` 覆盖：

```bash
WORK_DIR=/data/ops bash d.ops_develop/b.image_container/run.sh
```

生成的 `start_container.sh` 是**当前机器专用**模板；同一台机器可反复使用，换机器应重新运行。

### 3.3 ③ 容器内环境检查（只读，只给建议）

启动容器后进入容器：

```bash
docker exec -it asc_dev bash
```

然后在容器内运行：

```bash
bash d.ops_develop/c.env_check/run.sh
```

脚本在**当前容器环境**检查：

- 芯片型号：优先 `npu-smi info`
- 软件版本：`Python` ↔ `torch` ↔ `torch_npu` ↔ `CANN` ↔ 芯片型号
- CANN 安装目录和 `set_env.sh` 激活脚本

该脚本只提示，不执行：

- 缺 CANN / 版本不符：提示进入 `d.install_cann` 选版本补装
- 检查通过：提示进入 `e.op_design`

### 3.4 ④ 按需安装/补装 CANN toolkit（容器内）

> 仅当环境检查报告建议“安装/更换 CANN”时再执行；检查通过可跳过。

```bash
# 推荐 9.1.0
bash d.ops_develop/d.install_cann/a.cann-9.1.0/run.sh

# 9.0.0
bash d.ops_develop/d.install_cann/b.cann-9.0.0/run.sh
```

版本目录：

| 目录 | CANN 版本 | 说明 |
|---|---|---|
| `a.cann-9.1.0/` | `9.1.0` | 推荐 |
| `b.cann-9.0.0/` | `9.0.0` | 稳定 |
| `c.cann-8.2.RC1/` | `8.2.RC1` | 旧芯片兼容 |
| `d.cann-8.1.RC1/` | `8.1.RC1` | 旧芯片兼容 |

容器内默认安装到：

```text
/workspace/Ascend/ascend-toolkit-<version>
```

### 3.5 ⑤ 算子需求分析 → 生成 op.json

```bash
bash d.ops_develop/e.op_design/run.sh
```

运行后可选：

- `[1] / [2]` 大模型分析：从 `a.llm_config` 创建的配置中选择
- `[3]` 手动填写

生成（默认在 `d.ops_develop/workspace/` 下归档）：

```text
d.ops_develop/workspace/op_design_<算子名>/
├── op.json
└── op_spec.md
```

### 3.6 ⑥ 生成算子工程

```bash
# 自动在 d.ops_develop/workspace/ 下查找 op.json
bash d.ops_develop/f.op_build/run.sh

# 或明确指定 op.json
bash d.ops_develop/f.op_build/run.sh d.ops_develop/workspace/op_design_MatMulCustom/op.json
```

工程固定用 AscendC/C++ 模板生成，命令等价于：

```bash
msopgen gen -i <op.json> -f tf -lan cpp -c <compute_unit> -out <输出目录>
```

输出工程默认在：

```text
d.ops_develop/workspace/op_build_<算子名>/
```

### 3.7 ⑦ 大模型辅助修改算子工程

```bash
bash d.ops_develop/g.op_fix/run.sh
```

脚本会：

1. 列出 `workspace/` 下的算子工程
2. 从 `a.llm_config` 的配置中选择一个
3. 进入多轮对话修改模式
4. 保持上下文记忆

上下文按工程保存：

```text
workspace/op_build_<算子名>/.itool/op_fix_history.jsonl
```

### 3.8 ⑧ 安装 OpenCode + CANNBot skills/agents

```bash
# 解压并启动 opencode（便携模式）
bash d.ops_develop/h.op_agent/a.install_opencode/run.sh

# 安装 opencode + CANNBot skills/agents 到用户配置目录
bash d.ops_develop/h.op_agent/b.install_cannbot_skill/run.sh
```

绿色包内置：

```text
skills : 74 个
agents : 18 个
版本   : CANNBot 1.1.0
```

非 root 用户默认安装到 `$HOME/.local/bin/opencode`，配置目录为 `${XDG_CONFIG_HOME:-$HOME/.config}/opencode`。

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
