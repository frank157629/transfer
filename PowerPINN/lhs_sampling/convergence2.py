# -*- coding: utf-8 -*-
"""
convergence2.py
###FOR EVALUATION###
从 *dense* 数据集中筛选：按 δ 尾窗是否收敛到 k·2π+Δ0 分三类采样，并保存到 lhs_sampling/（文件名含 dense）
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ========= 配置区（按需改） =========
# 你的 dense 数据集路径（相对本脚本，或给绝对路径）
dataset_id         = 1
data_path          = f"../data/data_dense/GFL_2nd_order/dataset_v{dataset_id}_dense.pkl"

# 判定参数
tail_window        = 100            # 尾窗长度（最后多少个采样点）
delta_offset       = 0.137          # Δ0（Rahul 模型的平衡点偏移）
delta_tol          = 0.001           # 尾窗内 |δ - (k·2π+Δ0)| ≤ tol 全满足才判为收敛

# 采样设置（目标比例，实际会根据可用数量自适应补齐）
sampling_ratio = dict(
    region_0      = 1,   # 收敛到 0·2π+Δ0
    region_2kpi   = 0,   # 收敛到 k≠0 的 2kπ+Δ0
    non_converged = 0.0    # 非收敛
)
num_points_to_save = 1000  # 保存总条数

# 可视化
plot_quick_examples = True  # 是否画三类轨迹的快速浏览图
save_convergence_map = True # 是否保存初值散点图

# 输出目录（设为脚本同目录的 lhs_sampling）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
out_dir    = SCRIPT_DIR
# =================================


pi = np.pi

def load_dataset(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    # 期待格式：list/array，形状 (N, 3, T) -> [t, delta, omega]
    data = np.array(data, dtype=np.float64)
    assert data.ndim == 3 and data.shape[1] == 3, f"期望 (N,3,T)，得到 {data.shape}"
    return data

def classify_by_delta_tail(data: np.ndarray,
                           tail: int,
                           offset: float,
                           tol: float):
    """
    只看 δ 尾窗收敛：
      center = k·2π + offset
      若尾窗内所有 δ ∈ [center ± tol] -> 记入该 k
    返回：dict{k: [traj_id, ...]}
    """
    groups = defaultdict(list)
    N      = data.shape[0]

    for idx in range(N):
        delta_tail = data[idx, 1, -tail:]
        mean_tail  = np.mean(delta_tail)
        k          = int(np.round((mean_tail - offset) / (2*pi)))
        center     = k * 2 * pi + offset
        if np.all(np.abs(delta_tail - center) <= tol):
            groups[k].append(idx)
    return groups

def plot_ic_map(groups, data, tag="dense"):
    """初值散点（δ0, ω0）：k=0（绿）、k≠0（蓝）、非收敛（红）"""
    region0, regionK, nonconv = [], [], []
    all_ids  = set(range(data.shape[0]))
    conv_ids = set()

    for k, lst in groups.items():
        conv_ids.update(lst)
        for idx in lst:
            region = region0 if k == 0 else regionK
            region.append((data[idx, 1, 0], data[idx, 2, 0]))
    for idx in (all_ids - conv_ids):
        nonconv.append((data[idx, 1, 0], data[idx, 2, 0]))

    def arr(x): return np.array(x) if x else np.empty((0,2))
    A, B, C = map(arr, (region0, regionK, nonconv))

    plt.figure(figsize=(7.5,6))
    if A.size: plt.scatter(A[:,0], A[:,1], s=2, c='green', label=f"k=0  ({len(A)})")
    if B.size: plt.scatter(B[:,0], B[:,1], s=2, c='blue',  label=f"k≠0 ({len(B)})")
    if C.size: plt.scatter(C[:,0], C[:,1], s=2, c='red',   label=f"non-conv ({len(C)})")
    plt.xlabel("δ0"); plt.ylabel("ω0"); plt.title(f"IC scatter (dense)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    out_pdf = os.path.join(out_dir, f"convergence_map_dense_v{dataset_id}.pdf")
    plt.savefig(out_pdf, dpi=160); plt.close()
    print(f"🖼  IC map saved → {out_pdf}")

def quick_plot_group(data, ids, title, max_show=200):
    """把某一组的 δ/ω 曲线快速画出来（最多 max_show 条，免得太挤）"""
    if not ids:
        print(f"[{title}] 空集，跳过。")
        return
    ids = ids[:max_show]
    T = data.shape[2]
    t = data[0, 0, :]  # 用第一条的时间轴

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for idx in ids:
        axes[0].plot(t, data[idx, 1], lw=0.8)
    axes[0].set_ylabel('δ (rad)'); axes[0].set_title(title); axes[0].grid(alpha=.3)

    for idx in ids:
        axes[1].plot(t, data[idx, 2], lw=0.8)
    axes[1].set_ylabel('ω'); axes[1].set_xlabel('time (s)'); axes[1].grid(alpha=.3)

    fig.tight_layout(); plt.show()

def allocate_samples(k0_ids, kN_ids, non_ids, total, ratios, seed=42):
    """
    按比例 & 可用数量自适应分配采样条数。
    """
    rng = np.random.default_rng(seed)

    # 目标配额（向下取整）
    r0  = max(0.0, float(ratios.get("region_0", 0.0)))
    rK  = max(0.0, float(ratios.get("region_2kpi", 0.0)))
    rN  = max(0.0, float(ratios.get("non_converged", 0.0)))
    rs  = np.array([r0, rK, rN], dtype=float)
    if rs.sum() == 0:
        rs = np.array([1.0, 0.0, 0.0])  # 全给 k=0
    rs /= rs.sum()

    target = np.floor(rs * total).astype(int)
    # 把由于取整丢的名额补上
    while target.sum() < total:
        idx = np.argmax(rs - target/total)   # 简单补偿策略
        target[idx] += 1

    pools  = [k0_ids, kN_ids, non_ids]
    picks  = []
    taken  = np.zeros(3, dtype=int)

    # 先按目标拿
    for i, pool in enumerate(pools):
        n = min(len(pool), target[i])
        if n > 0:
            picks.append(rng.choice(pool, n, replace=False))
        else:
            picks.append(np.array([], dtype=int))
        taken[i] = n

    # 有的类不够 -> 把剩余额度分配给其它类
    leftover = total - taken.sum()
    if leftover > 0:
        avail_counts = [len(pools[i]) - taken[i] for i in range(3)]
        order = np.argsort(-rs)  # 优先把名额给比例大的类
        for i in order:
            if leftover <= 0: break
            can = max(0, avail_counts[i])
            add = min(can, leftover)
            if add > 0:
                rest_pool = np.setdiff1d(np.array(pools[i], dtype=int), picks[i], assume_unique=False)
                if rest_pool.size:
                    add_ids = rng.choice(rest_pool, add, replace=False)
                    picks[i] = np.concatenate([picks[i], add_ids])
                    leftover -= add

    # 合并 & 打乱
    all_ids = np.concatenate([*picks]) if any(len(p) for p in picks) else np.array([], dtype=int)
    rng.shuffle(all_ids)
    return all_ids.astype(np.int64), taken

# ----------------- 主流程 -----------------
if __name__ == "__main__":
    print(f"📥 Loading dense dataset: {data_path}")
    data = load_dataset(data_path)  # (N,3,T)

    print("🧮 Classifying by δ tail behaviour …")
    groups = classify_by_delta_tail(data, tail_window, delta_offset, delta_tol)

    # 统计
    total_conv = sum(len(v) for v in groups.values())
    k0_ids  = groups.get(0, [])
    kN_ids  = [idx for k, lst in groups.items() if k != 0 for idx in lst]
    non_ids = list(set(range(data.shape[0])) - set(k0_ids) - set(kN_ids))

    print(f"📊 total trajectories            : {data.shape[0]}")
    print(f"📊 converged (all k)             : {total_conv}")
    print(f"    ├─ k = 0                     : {len(k0_ids)}")
    print(f"    └─ k ≠ 0 (2kπ+Δ0)            : {len(kN_ids)}")
    print(f"📊 non-converged                 : {len(non_ids)}")

    # 初值散点
    if save_convergence_map:
        plot_ic_map(groups, data, tag="dense")

    # 采样
    sample_ids, taken = allocate_samples(k0_ids, kN_ids, non_ids,
                                         num_points_to_save, sampling_ratio, seed=42)
    n0, nK, nN = taken.tolist()
    print(f"🎯 target total {num_points_to_save} → sampled: k=0[{n0}], k≠0[{nK}], non[{nN}] → sum={len(sample_ids)}")

    # 可选：快速看一下三类曲线
    if plot_quick_examples:
        quick_plot_group(data, k0_ids,  "Class A  (δ → 0·2π+Δ0)")
        quick_plot_group(data, kN_ids,  "Class B  (δ → k·2π+Δ0, k≠0)")
        quick_plot_group(data, non_ids, "Class C  (Non-converged)")

    # 保存
    final_ds  = data[sample_ids]
    save_name = f"dataset_v{dataset_id}_dense_mixed_k{len(sample_ids)}.pkl"
    save_path = os.path.join(out_dir, save_name)
    with open(save_path, "wb") as f:
        pickle.dump(final_ds, f)
    print(f"✅ Saved {len(sample_ids)} trajectories → {save_path}")

    # 保存初值散点（已采样子集）
    d0 = final_ds[:, 1, 0]
    w0 = final_ds[:, 2, 0]
    plt.figure(figsize=(7.5,6))
    plt.scatter(d0, w0, s=2, c='black', alpha=.6)
    plt.xlabel("δ0"); plt.ylabel("ω0"); plt.title("IC of saved (dense) trajectories")
    plt.grid(True); plt.tight_layout()
    pdf_path = os.path.join(out_dir, f"ic_map_dataset_v{dataset_id}_dense_mixed_k{len(sample_ids)}.pdf")
    plt.savefig(pdf_path, dpi=160); plt.close()
    print(f"🖼  IC map (saved subset) → {pdf_path}")