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
#   FOLDER	a.env_check	a
#   FOLDER	b.env_setup	b
#   ...

# 打包下载 + 预览
curl -s -H 'Authorization: Bearer <token>' -o pack.tar.gz 'http://<server-A>:5170/api/pack?path=e.environment/a.versions/a.all'
curl -s -H 'Authorization: Bearer <token>' 'http://<server-A>:5170/api/cat?path=e.environment/a.versions/a.all'
```

---

## 3. 算子开发工作流（d.ops_develop）

面向：在客户机器上开发算子 / 基于算子源码改造。环境准备 ①→④ + 开发两步 ⑤→⑥，均可独立执行。环境检查已合并为单脚本，宿主机与容器通用。

### 3.1 ① 环境检查（单脚本：芯片型号 + 软件版本匹配矩阵）

```bash
bash d.ops_develop/a.env_check/run.sh
```

该脚本在**当前 shell 所在环境**直接检查（不做交互选择，不修改系统），输出重点：

```
════════════════════════════════════════════════════════════
  【环境检查汇总报告】
════════════════════════════════════════════════════════════
  运行位置    : 宿主机 / 容器内
  芯片型号    : 910B (910B)
  识别来源    : npu-smi info
  NPU 设备数  : 8
  Python      : 3.10.13
  CANN        : 8.1.RC1
  安装目录    : /usr/local/Ascend/ascend-toolkit/latest
  激活脚本    : /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  torch       : 2.1.0
  torch_npu   : 2.1.0.post16
────────────────────────────────────────────────────────────
  ✅ 综合结论: 当前环境软件版本匹配, 可以继续算子开发。
```

兼容性判断要点：
- `Python` ↔ `torch` 常见支持范围；
- `torch` ↔ `torch_npu` 主版本是否同系列；
- `芯片型号` ↔ `CANN` ↔ `torch_npu` 是否命中常见配套（910 系列 / 950 / 310P）。

该脚本仅检查，**不修复、不安装、不自动写环境变量**。发现缺口时按需执行下面脚本：
- 缺 CANN / 版本不符：进入 `a.install_cann` 选择版本安装（见 3.2），推荐 `a.cann-9.1.0`
- 缺 torch/torch_npu：参考 `e.environment/e.setenvs/setenvs.sh` 中的 `install_torch`（当前内置支持 2.1.0 / 2.6.0）
- 镜像 / 容器：见 3.3 / 3.4；进入容器后再次运行本脚本确认容器内版本也匹配。

### 3.2 ② CANN toolkit 安装（下载 + 安装合并，仅 toolkit，可选择版本目录）

进入 `a.install_cann` 后，按 `a` / `b` / `c` / `d` 选择版本；也可直接运行对应版本脚本：

```bash
# 推荐：9.1.0
bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 9.0.0
bash d.ops_develop/b.env_setup/a.install_cann/b.cann-9.0.0/run.sh

# 旧芯片兼容
bash d.ops_develop/b.env_setup/a.install_cann/c.cann-8.2.RC1/run.sh
bash d.ops_develop/b.env_setup/a.install_cann/d.cann-8.1.RC1/run.sh
```

每个版本只安装 `Ascend-cann-toolkit_<version>_linux-<arch>.run`，不安装 kernels/ops、不安装合一包/驱动。

- CPU 架构自动识别：`uname -m` → `lscpu` → `dpkg --print-architecture`，归一化为 `x86_64` 或 `aarch64`。
- 安装目录防覆盖：默认建议版本化新目录；若目标目录已存在 CANN，默认拒绝覆盖，必须显式 `ITOOL_FORCE=1` 才允许继续。

自动流程：查包/缺包询问下载 → 下载 toolkit → `.run --install --install-path=<安装目录>` → 自动 `source set_env.sh` → `python3 -c "import acl"` 验证 → 可选写入 `~/.bashrc`。

常用非交互/自动化变量：

```bash
# 只检查官网 URL 是否可达, 不下载/不安装
CHECK_ONLY=1 bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 自动下载 + 自动安装
ITOOL_AUTO_DL=1 ITOOL_AUTO_INSTALL=1 bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh

# 指定安装目录 / 包目录
INSTALL_DIR=/opt/Ascend PKG_DIR=/opt/cann_pkgs bash d.ops_develop/b.env_setup/a.install_cann/b.cann-9.0.0/run.sh

# 内网环境换源: 保留 __VER__ 占位符
CANN_BASE_URL='https://内网镜像/CANN/CANN%20__VER__' bash d.ops_develop/b.env_setup/a.install_cann/a.cann-9.1.0/run.sh
```

> 当前已按官网资源探测可用的 toolkit 版本：`9.1.0`、`9.0.0`、`8.2.RC1`、`8.1.RC1`。

### 3.3 ③ 镜像拉取 + 容器实例化（合并）

```bash
bash d.ops_develop/c.container/run.sh
```

该脚本先完成镜像选择/拉取，再自动进入容器实例化流程。

**镜像阶段**：自动查询 `quay.io/ascend/cann` 全部 tag → 按 芯片/版本/系统/Python 筛选 → 编号列表选择：

```
按需筛选(直接回车=不限):
  芯片(910b/910a/950/310p, 留空=全部) []: 910b
  CANN 版本(如 8.1.rc1 / 9.0.0, 留空=全部) []:
  系统(如 ubuntu22.04 / openeuler22.03, 留空=全部) []:
  Python(如 py3.10 / py3.11, 留空=全部) []:

════════ 匹配的镜像 tag (3) ════════
    1) 8.1.rc1-910b-ubuntu22.04-py3.10
    2) 8.1.rc1-910b-ubuntu24.04-py3.10
    3) 9.0.0-910b-ubuntu22.04-py3.10
────────────────────────────────────
  选择编号 [3]:
```

拉取后自动打本地短标签 `cann-910b:9.0.0`。也可直接指定：

```bash
IMAGE=quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10 bash d.ops_develop/c.container/run.sh
```

> 网络健壮性：脚本会依次尝试 quay.io 官方 API / Docker Registry v2 API 并自动重试；若都失败，会给出兜底选项——`[1] 重试` / `[2] 用内置常见 tag 列表` / `[3] 手动输入镜像`，无需手动排查。

**容器阶段**：收集容器名/工作目录/共享内存，自动枚举 `/dev/davinci*` 设备，生成 **`start_container.sh`** 并立即启动：

```
容器名 [asc_dev]:
工作目录(映射到容器 /workspace) [/data/ops]:
共享内存(--shm-size, 如 16g) [16g]:
✔ 已生成起容器脚本: /data/ops/start_container.sh
```

改完直接 `bash start_container.sh` 即可重建容器；进入容器后重新执行环境检查（3.1）：

```bash
docker exec -it asc_dev bash
# 容器内执行:
bash d.ops_develop/a.env_check/run.sh
```

### 3.4 ④ 算子需求分析 → 生成 op.json

```bash
bash d.ops_develop/d.design/a.op_spec/run.sh
```

交互示例（回车用默认值）：

```
算子名称 [AddCustom]: MatMulCustom
算子类型(elementwise/matmul/reduce/custom) [custom]: matmul
算子功能描述: 矩阵乘法 A x B
支持数据类型(如 fp16,fp32,int8) [fp16,fp32]: fp16
输入数量 [2]: 2
  --- 输入 1 ---
    名称 [x1]: A
    数据类型 [fp16]: fp16
    典型shape [ -1,-1 ]: 1024,1024
  ...
```

生成 `op_design_MatMulCustom/op.json`（msopgen 格式）和 `op_spec.md`。

### 3.5 ⑤ 生成算子工程 / 接入

```bash
# 轻量: msopgen 生成 AscendC 工程
bash d.ops_develop/e.scaffold/a.msopgen/run.sh op_design_MatMulCustom/op.json

# 完善: 拉取 ops-transformer 算子库(官方 gitcode)
bash d.ops_develop/e.scaffold/b.ops_transformer/run.sh

# 接入: torchbind(CPU + NPU) 工程
bash d.ops_develop/e.scaffold/c.torchbind/run.sh MatMulCustom
# 产物: MatMulCustom.cpp / setup.py / MatMulCustom_npu.cpp / setup_npu.py / README.md
```

编译：

```bash
cd torchbind_MatMulCustom
python setup.py install          # CPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
