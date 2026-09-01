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

面向：在客户机器上开发算子 / 基于算子源码改造。四步可独立执行。

### 3.1 环境检查

```bash
bash d.ops_develop/a.env_check/a.check_env/run.sh
```

输出会先打印环境变量，再逐项检查组件并给建议：

```
===== 组件检查 (算子开发依赖) =====
  [ OK ] TOOLKIT (ascend-toolkit)  路径: /usr/local/Ascend/ascend-toolkit/latest
  [缺失] OPP (算子原型库)
          建议: 设置 ASCEND_OPP_PATH 指向 <toolkit>/opp
  [缺失] torch
          建议: pip install torch (昇腾环境需配套 torch/torch_npu 版本)
  [ OK ] cmake   cmake version 3.31.3
  [ OK ] gcc     gcc (GCC) 11.4.0
```

### 3.2 下载 CANN（先探测，再正式下载）

```bash
# 只探测 URL 是否可达(不下载大包)
CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 CHECK_ONLY=1 \
  bash d.ops_develop/a.env_check/b.download_cann/run.sh

# 正式下载(toolkit 约 2.2G + kernel 约 1.9G, 支持断点续传)
CANN_VERSION=8.1.RC1 CHIP=910b ARCH=x86_64 \
  bash d.ops_develop/a.env_check/b.download_cann/run.sh
```

内网环境可换源：`CANN_BASE_URL=https://内网镜像/CANN/...`。

### 3.3 拉镜像 + 起容器

```bash
# 镜像 tag 形如 8.1.rc1-910b-ubuntu22.04-py3.10
CANN_VERSION=8.1.rc1 CHIP=910b bash d.ops_develop/b.env_setup/a.pull_vllm_ascend/run.sh

# 实例化(自动枚举 /dev/davinci* 挂载)
bash d.ops_develop/b.env_setup/b.run_container/run.sh cann-910b:8.1.rc1 my_ascend
docker exec -it my_ascend bash
```

### 3.4 算子需求分析 → 生成 op.json

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

### 3.5 生成算子工程 / 接入

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
