import numpy as np
import pandas as pd

# ====================== 配置 ======================
FILE_A = "layer_outputs_modelA.npz"
FILE_B = "layer_outputs_modelB.npz"
REPORT_CSV = "vllm_alignment_report.csv"
ATOL = 1e-5  # 绝对误差阈值
RTOL = 1e-3  # 相对误差阈值
# ===================================================

def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def compare_npz(file_a, file_b):
    data_a = np.load(file_a)
    data_b = np.load(file_b)
    
    layers = sorted(data_a.files, key=lambda x: int(x.split("_")[-1]))
    records = []

    for layer in layers:
        arr_a = data_a[layer]
        arr_b = data_b[layer]

        # 核心对比指标
        abs_err = np.mean(np.abs(arr_a - arr_b))
        max_abs_err = np.max(np.abs(arr_a - arr_b))
        cos_sim = cosine_similarity(arr_a, arr_b)
        is_aligned = np.allclose(arr_a, arr_b, atol=ATOL, rtol=RTOL)

        records.append({
            "layer": layer,
            "mean_abs_error": round(abs_err, 8),
            "max_abs_error": round(max_abs_err, 8),
            "cosine_similarity": round(cos_sim, 8),
            f"aligned(atol={ATOL})": is_aligned
        })

    df = pd.DataFrame(records)
    return df

# 运行对比
report = compare_npz(FILE_A, FILE_B)

# 输出报告
print("\n===== vLLM 模型多层精度对齐报告 =====")
print(report.to_string(index=False))
report.to_csv(REPORT_CSV, index=False)

# 未对齐层统计
unaligned = report[~report.iloc[:, -1]]
print(f"\n未对齐层总数：{len(unaligned)}")
if not unaligned.empty:
    print("\n未对齐层详情：")
    print(unaligned.to_string(index=False))
