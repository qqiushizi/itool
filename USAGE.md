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
#   FOLDER	a.image_container	a
#   FOLDER	b.env_check	b
#   FOLDER	c.install_cann	c
#   ...

# 打包下载 + 预览
curl -s -H 'Authorization: Bearer <token>' -o pack.tar.gz 'http://<server-A>:5170/api/pack?path=e.environment/a.versions/a.all'
curl -s -H 'Authorization: Bearer <token>' 'http://<server-A>:5170/api/cat?path=e.environment/a.versions/a.all'
```

---

## 3. 算子开发工作流（d.ops_develop）

面向：在客户机器上开发算子 / 基于算子源码改造。工作流先拉镜像、建容器，后续检查与开发均在容器内完成。每步都是独立 `run.sh`。

```
① a.image_container  拉镜像 + 建容器
② b.env_check        容器内环境检查（只读，只给建议）
③ c.install_cann     按需安装/补装 CANN toolkit（检查通过可跳过）
④ d.op_design        需求分析 → op.json + op_spec.md
⑤ e.op_scaffold      生成算子工程 / 接入
```

### 3.1 ① 镜像拉取 + 容器实例化

```bash
bash d.ops_develop/a.image_container/run.sh
```

脚本会询问镜像地址和容器名称（默认 `asc_dev`）。也支持非交互：

```bash
IMAGE=quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 bash d.ops_develop/a.image_container/run.sh

bash d.ops_develop/a.image_container/run.sh quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.10 asc_dev
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
WORK_DIR=/data/ops bash d.ops_develop/a.image_container/run.sh
```

生成的 `start_container.sh` 是**当前机器专用**模板；同一台机器反复重启容器可直接使用，换机器应重新运行 `a.image_container`。
### 3.2 ② 容器内环境检查（只读，只给建议）

启动容器后进入容器：

```bash
docker exec -it asc_dev bash
```

然后在容器内运行：

```bash
bash d.ops_develop/b.env_check/run.sh
```

脚本在**当前容器环境**检查：

- 芯片型号：优先 `npu-smi info`，识别 910B / 910A / 910C / 950 / 310P 等。
- 软件版本：`Python` ↔ `torch` ↔ `torch_npu` ↔ `CANN` ↔ 芯片型号。
- CANN 安装目录和 `set_env.sh` 激活脚本。

输出示例：

```
运行位置    : 容器内
芯片型号    : 910B
Python      : 3.10.13
CANN        : 9.0.0
安装目录    : /workspace/Ascend/ascend-toolkit-9.0.0
激活脚本    : /workspace/Ascend/ascend-toolkit-9.0.0/set_env.sh
torch       : 2.6.0
torch_npu   : 2.6.0.post2

✅ 综合结论: 当前环境软件版本匹配, 可以继续算子开发。
```

该脚本只提示，不执行：

- 缺 CANN / 版本不符：提示进入 `c.install_cann` 选版本补装。
- 缺 torch / torch_npu：提示参考 `e.environment/e.setenvs/setenvs.sh` 的 `install_torch`。
- 检查通过：提示进入 `d.op_design`。

### 3.3 ③ 按需安装/补装 CANN toolkit（容器内）

> 仅当 3.2 的报告建议“安装/更换 CANN”时再执行；环境检查通过可跳过。

```bash
# 推荐 9.1.0
bash d.ops_develop/c.install_cann/a.cann-9.1.0/run.sh

# 9.0.0
bash d.ops_develop/c.install_cann/b.cann-9.0.0/run.sh
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

`/workspace` 由 `a.image_container` 挂载宿主机工作目录，所以容器删除/重建后安装结果仍然保留。

脚本自动识别 `x86_64` / `aarch64`，然后：查包/下载 toolkit → 安装 → `source set_env.sh` → `import acl` 验证。未进入容器直接运行会退出；宿主机只探测 URL 可加 `ITOOL_ALLOW_HOST=1 CHECK_ONLY=1`。

### 3.4 ④ 算子需求分析 → 生成 op.json

```bash
bash d.ops_develop/d.op_design/a.analysis/run.sh
```

运行后可选：

- `[1] 大模型分析`
- `[2] 手动填写`

大模型模式支持外部 API 和使用工具的昇腾宿主机上已启动的 vLLM 服务，统走 OpenAI Chat Completions 协议：

```bash
# 昇腾宿主机上的 vLLM-ascend 服务
ITOOL_LLM_API_BASE=http://127.0.0.1:8000/v1 ITOOL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct bash d.ops_develop/d.op_design/a.analysis/run.sh

# 外部 API
ITOOL_LLM_API_BASE=https://api.deepseek.com/v1 ITOOL_LLM_API_KEY=sk-xxxx ITOOL_LLM_MODEL=deepseek-chat bash d.ops_develop/d.op_design/a.analysis/run.sh
```

生成：

- `op_design_<算子名>/op.json`
- `op_spec.md`

### 3.5 ⑤ 生成算子工程 / 接入

```bash
# msopgen AscendC 工程
bash d.ops_develop/e.op_scaffold/a.msopgen/run.sh op_design_MatMulCustom/op.json

# ops-transformer 源码
bash d.ops_develop/e.op_scaffold/b.ops_transformer/run.sh

# torchbind(CPU + NPU)
bash d.ops_develop/e.op_scaffold/c.torchbind/run.sh MatMulCustom
```

编译示例：

```bash
cd torchbind_MatMulCustom
source /workspace/Ascend/ascend-toolkit-9.1.0/set_env.sh
python setup.py install          # CPU
python setup_npu.py install      # NPU
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
