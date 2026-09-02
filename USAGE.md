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

面向：在客户机器上开发算子 / 基于算子源码改造。环境准备五步 ①→⑤ + 开发两步 ⑥→⑦，均可独立执行。

### 3.1 ① 服务器 CANN 检查（安装目录/版本/驱动/激活）

```bash
bash d.ops_develop/a.env_check/a.check_cann/run.sh
```

汇总输出示例（检查主机/OS、驱动 HDK、CANN 安装目录、toolkit/kernels 版本、环境变量激活状态、Python/框架、Docker）：

```
════════════════════════════════════════════════════
  【汇总报告】
════════════════════════════════════════════════════
  安装目录    : /usr/local/Ascend/ascend-toolkit/latest
  CANN 版本   : version=8.1.RC1
  激活脚本    : /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  当前激活状态: 已找到 set_env.sh
  驱动信息    : driver version = 25.0.1
  NPU 设备数  : 8
────────────────────────────────────────────────────
  下一步建议: 环境基本就绪 → ③ 拉取镜像 → ④ 起容器 → ⑤ 进容器检查
```

检查完会自动进入**一键修复向导**，按检测到的缺失项列出可选方案，你只需选编号（或 `A` 全部）：

```
════════════════════════════════════════════════════
  【可选修复方案】(选择编号执行, 无需手动检查)
════════════════════════════════════════════════════
    [1] 安装 CANN (当前缺失)
    [2] 生成/激活 CANN 环境脚本 activate_cann.sh
    [3] 安装 torch + torch_npu
    [4] 安装 pybind11
    [5] 安装编译工具链 gcc/cmake
    [6] Docker 安装指引
    [A] 一键修复以上全部缺失项
    [0] 跳过(仅查看)

  请选择方案 [A]:
```

- 选 `A`：按顺序自动补齐所有缺失项
- 选 `2`：生成 `activate_cann.sh`（内容即 `source <set_env.sh>`），可写入 `~/.bashrc` 永久激活，也可手动 `source activate_cann.sh`

### 3.2 下载 CANN 包（可选，先探测再下载）

```bash
# 只探测 URL 是否可达(不下载大包)
CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 CHECK_ONLY=1 \
  bash d.ops_develop/a.env_check/b.download_cann/run.sh

# 正式下载(默认 toolkit + kernels, 支持断点续传)
CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 \
  bash d.ops_develop/a.env_check/b.download_cann/run.sh

# 仅 toolkit / 合一包
MODE=toolkit  CANN_VERSION=8.1.RC1 bash d.ops_develop/a.env_check/b.download_cann/run.sh
MODE=combined CANN_VERSION=8.1.RC1 bash d.ops_develop/a.env_check/b.download_cann/run.sh
```

内网环境可换源：`CANN_BASE_URL=https://内网镜像/CANN/...`。

### 3.3 ② CANN 安装（交互：方式 / 位置 / source 激活）

```bash
bash d.ops_develop/a.env_check/c.install_cann/run.sh
```

交互选择：

```
请选择安装方式:
  1) toolkit + kernels(ops)   [算子开发推荐]
  2) 仅 toolkit
  3) 合一包 (驱动 + toolkit + 其他, 单个 .run)
选择 [1]:
CANN 版本 [8.1.RC1]:
安装位置 [/usr/local/Ascend/ascend-toolkit]:
```

脚本会：定位/自动下载安装包 → 执行 `.run` 安装(非 root 自动加 sudo) → `source <安装目录>/set_env.sh` 激活 → `python3 -c "import acl"` 验证 → 可选写入 `~/.bashrc` 永久激活。

### 3.4 ③ 镜像拉取（quay.io 可视化选 tag）

```bash
bash d.ops_develop/b.env_setup/a.pull_image/run.sh
```

交互流程：自动查询 `quay.io/ascend/cann` 全部 tag → 按 芯片/版本/系统/Python 筛选 → 编号列表选择：

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

拉取后自动打本地短标签 `cann-910b:9.0.0`。也可直接指定：`IMAGE=quay.io/ascend/cann:8.1.rc1-910b-ubuntu22.04-py3.10 bash .../a.pull_image/run.sh`。

> 网络健壮性：脚本会依次尝试 quay.io 官方 API / Docker Registry v2 API 并自动重试；若都失败，会给出兜底选项——`[1] 重试` / `[2] 用内置常见 tag 列表` / `[3] 手动输入镜像`，无需手动排查。

### 3.5 ④ 容器实例化（交互 + 生成可编辑起容器脚本）

```bash
bash d.ops_develop/b.env_setup/b.run_container/run.sh
```

交互收集镜像/容器名/工作目录/共享内存，自动枚举 `/dev/davinci*` 设备，并生成 **`start_container.sh`**（客户可自行修改后重复执行），随后立即启动容器：

```
镜像 [cann-910b:8.1.rc1]:
容器名 [asc_dev]:
工作目录(映射到容器 /workspace) [/data/ops]:
共享内存(--shm-size, 如 16g) [16g]:
✔ 已生成起容器脚本: /data/ops/start_container.sh
```

改完直接 `bash start_container.sh` 即可重建容器；进入：`docker exec -it asc_dev bash`。

### 3.6 ⑤ 进容器检查软件包 → 确认可开始算子开发

```bash
bash d.ops_develop/b.env_setup/c.check_in_container/run.sh asc_dev
```

在容器内检查 CANN/torch/torch_npu/编译链/NPU 设备，并给出结论：

```
===== 6. 结论 =====
  ✅ GO: 容器内软件包齐全, 可以开始算子开发。
  建议下一步:
    · 算子需求分析:  bash d.ops_develop/c.design/a.op_spec/run.sh
    · 生成工程:      bash d.ops_develop/d.scaffold/a.msopgen/run.sh
```

### 3.7 ⑥ 算子需求分析 → 生成 op.json

```bash
bash d.ops_develop/c.design/a.op_spec/run.sh
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

### 3.8 ⑦ 生成算子工程 / 接入

```bash
# 轻量: msopgen 生成 AscendC 工程
bash d.ops_develop/d.scaffold/a.msopgen/run.sh op_design_MatMulCustom/op.json

# 完善: 拉取 ops-transformer 算子库(官方 gitcode)
bash d.ops_develop/d.scaffold/b.ops_transformer/run.sh

# 接入: torchbind(CPU + NPU) 工程
bash d.ops_develop/d.scaffold/c.torchbind/run.sh MatMulCustom
# 产物: MatMulCustom.cpp / setup.py / MatMulCustom_npu.cpp / setup_npu.py / README.md
```

编译：

```bash
cd torchbind_MatMulCustom
python setup.py install          # CPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup_npu.py install      # NPU
```

---

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
