---
name: a5-env-check
description: 昇腾 A5（Ascend 950PR/950DT）机器的型号确认与环境检查，通过 npu-smi 查询芯片型号、确认架构代际与 SocVersion。当用户需要确认某台机器是不是 A5、是 950PR 还是 950DT、以及如何配置对应的 SocVersion 时使用。
---

# A5 型号确认与环境检查

面向昇腾 **A5（Ascend 950 系列）** 机器的型号识别与环境检查。

## 如何确认机器是不是 A5

官方推荐通过 **npu-smi** 命令查询：

```bash
# 查看基本信息（含 Name）
npu-smi info

# 针对 Ascend 950PR / 950DT，查询 Chip Name 与 NPU Name
npu-smi info -t board -i <id>
```

其中 `id` 为设备 ID，可通过 `npu-smi info -l` 查出的 NPU ID 获取。

## 950 系列的两个型号

- **Ascend 950PR**：Prefill（预填充）场景
- **Ascend 950DT**：Decode（解码）场景

两者架构代际一致（DAV_3510），但用途不同，需在实际使用中区分。

## 架构代际与 SocVersion

- 架构代号（NpuArch）：**DAV_3510**
- SocVersion：**Ascend950**

作用：在 Ascend C 算子编译、msopgen 工程生成时，需要按 SocVersion 指定目标芯片。

## 与前几代芯片的对照

| 产品 | 架构代号 | SocVersion |
|---|---|---|
| Ascend910B 系列 | DAV_2201 | Ascend910B1~B4 / Ascend910B2C 等 |
| A5（950PR/950DT） | DAV_3510 | Ascend950 |

> 一个 NpuArch 可对应多个 SocVersion。对核内算子开发而言，通常用 NpuArch 区分即可。

## 固件与驱动获取说明

950 系列产品的固件与驱动为**受限获取**——昇腾社区下载页标注「当前 950 系列产品为受限获取，如您需要下载配套固件与驱动，请点击商用版获取」。

## 环境检查要点

1. 用 `npu-smi info` 确认机器是否为 950 系列、是 PR 还是 DT
2. 确认 CANN 版本（950 的 SIMT 等新能力自 CANN 9.1.0 起提供，具体以官方版本配套说明为准）
3. 算子开发时按 SocVersion = Ascend950 配置目标芯片
4. 固件驱动走商用版渠道获取

## 官方依据

本 skill 内容来自昇腾社区官方文档「单算子 API 调用」「固件与驱动」页面及官方 CANN 文档中关于芯片型号查询、架构代际的公开说明，未包含任何非官方或编造内容。