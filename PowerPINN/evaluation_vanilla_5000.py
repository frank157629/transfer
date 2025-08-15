# evaluation_pinn.py
import os, time, pickle, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==============================
# 一处配置（路径留白，自己粘贴）
# ==============================
CONFIG = {
    # "model_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/GFL_2nd_orderDynamicNN_1_750_4000000_1000_500000_None_None_1_0_0_0_Static_20250814-092028.pth",          # <- 填你的 .pth
    "model_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/vanilla_model_1000_5000_10000/GFL_2nd_orderDynamicNN_1_750_4000000_1000_500000_None_None_1_0_0_0_Static_20250814-134327.pth",
    "dataset_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/lhs_sampling/dataset_v8_mixed_k1000.pkl",        # <- 填 (N,3,T) 的测试集 [t,delta,omega]
    # "dataset_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/data/GFL_2nd_order/dataset_v11.pkl",
    "out_dir": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/vanilla/reports_vanilla_5000",   # <- PDF 输出目录
    "device": "auto",

    # 模型结构（与你训练一致）
    "input_size": 3,      # [t, delta0, omega0]
    "hidden_size": 128,
    "output_size": 2,     # [delta(t), omega(t)]
    "num_layers": 4,

    # 前向评测方式
    "one_shot": False,        # 大数据建议 False
    "batch_size": 131072,     # (N*T) 方向的 batch
}

# ==============================
# 模型结构（保持与训练一致）
# ==============================
class Network(nn.Module):
    """
    input:  [t, delta0, omega0]  -> dim=3
    output: [delta(t), omega(t)] -> dim=2
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers  = num_layers

        self.hidden = [nn.Linear(self.input_size, self.hidden_size)]
        for _ in range(self.num_layers):
            self.hidden.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        for i in range(self.num_layers):
            x = torch.tanh(self.hidden[i](x))
        return self.output(x)

# ==============================
# 工具函数
# ==============================
def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def load_checkpoint_strict(model: nn.Module, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print("[INFO] model loaded (strict=True).")

def load_dataset(path: str) -> np.ndarray:
    # 期望 (N,3,T) -> [t, delta, omega]
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = np.array(data, dtype=np.float32)
    assert data.ndim == 3 and data.shape[1] == 3, f"expect (N,3,T), got {data.shape}"
    return data

def make_xy_from_dataset(data: np.ndarray):
    """
    data: (N, 3, T)  [0:t, 1:delta(t), 2:omega(t)]
    x   : (N*T, 3)   [t, delta0, omega0]
    y   : (N*T, 2)   [delta(t), omega(t)]
    """
    N, _, T = data.shape
    t_all = data[:, 0, :]
    d_all = data[:, 1, :]
    w_all = data[:, 2, :]

    d0 = d_all[:, 0][:, None]
    w0 = w_all[:, 0][:, None]
    d0_rep = np.repeat(d0, T, axis=1)
    w0_rep = np.repeat(w0, T, axis=1)

    x_np = np.stack([t_all, d0_rep, w0_rep], axis=2).reshape(-1, 3).astype(np.float32)
    y_np = np.stack([d_all,  w_all ],        axis=2).reshape(-1, 2).astype(np.float32)
    return x_np, y_np, N, T

@torch.no_grad()
def forward_one_shot(model: nn.Module, x_np: np.ndarray, device: torch.device):
    x = torch.from_numpy(x_np).to(device, non_blocking=True)
    model.eval()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    y  = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    y_np = y.detach().cpu().numpy()
    return y_np, (t1 - t0)

@torch.no_grad()
def forward_in_batches(model: nn.Module, x_np: np.ndarray, device: torch.device, batch_size: int):
    model.eval()
    preds = []
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, x_np.shape[0], batch_size):
        xb = torch.from_numpy(x_np[i:i+batch_size]).to(device, non_blocking=True)
        yb = model(xb)
        preds.append(yb.detach().cpu().numpy())
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    y_np = np.concatenate(preds, axis=0)
    return y_np, (t1 - t0)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err  = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    maxe = float(np.max(np.abs(err)))
    return mae, mse, maxe

def error_over_time(y_true_flat, y_pred_flat, N, T):
    err = (y_pred_flat - y_true_flat).reshape(N, T, 2)  # (N,T,2)
    t_MAE  = np.mean(np.abs(err), axis=0)               # (T,2)
    t_MSE  = np.mean(err**2, axis=0)                    # (T,2)
    t_MaxAE = np.max(np.abs(err), axis=0)               # (T,2)
    return t_MAE, t_MSE, t_MaxAE

def save_two_lines(t, A, title, ylabel, out_pdf):
    """A: (T,2)，把 δ/ω 两条线画在同一张图"""
    plt.figure(figsize=(7,4))
    plt.plot(t, A[:,0], label="delta")
    plt.plot(t, A[:,1], label="omega")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def save_single_line(t, avg, title, ylabel, out_pdf):
    plt.figure(figsize=(7, 4))
    plt.plot(t, avg, label="avg (delta + omega)")
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def save_box_final(final_abs_delta, final_abs_omega, out_pdf):
    data = [
        np.array(final_abs_delta, dtype=np.float32),
        np.array(final_abs_omega, dtype=np.float32),
        0.5*(np.array(final_abs_delta)+np.array(final_abs_omega)),
        ]
    plt.figure(figsize=(6,4))
    plt.boxplot(data, tick_labels=["delta", "omega", "avg"])
    plt.ylabel("MAE at final time")
    plt.title("Final-time MAE distribution")
    plt.grid(True, ls="--", alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def pick_best_traj_idx(y_true_flat, y_pred_flat, N, T):
    """按整条轨迹的平均 MAE(两变量平均) 选最小的那条"""
    err = (y_pred_flat - y_true_flat).reshape(N, T, 2)
    mae_traj = np.mean(np.abs(err), axis=(1,2))  # (N,)
    return int(np.argmin(mae_traj)), mae_traj

def save_one_traj(t, y_true_flat, y_pred_flat, idx, N, T, out_pdf):
    y_true = y_true_flat.reshape(N, T, 2)[idx]
    y_pred = y_pred_flat.reshape(N, T, 2)[idx]
    plt.figure(figsize=(8,4))
    plt.plot(t, y_true[:,0], lw=1.2, label="δ true")
    plt.plot(t, y_pred[:,0], lw=1.2, ls='--', label="δ pred")
    plt.plot(t, y_true[:,1], lw=1.2, label="ω true")
    plt.plot(t, y_pred[:,1], lw=1.2, ls='--', label="ω pred")
    plt.xlabel("Time (s)")
    plt.ylabel("States")
    plt.title(f"Best trajectory (min test loss)  idx={idx+1}")
    plt.grid(ls="--", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def bench_like_paper(model, data_np, device, out_txt, ms=(1, 50, 500), batch_size=131072):
    """
    测量 forward 推理时间（单位：毫秒）
    参数：
        ms: 要测试的轨迹数量（列表）
        batch_size: 每次 forward 的 batch 大小
    """
    N, _, T = data_np.shape
    t_all = data_np[:, 0, :]
    d_all = data_np[:, 1, :]
    w_all = data_np[:, 2, :]

    lines = ["=== PURE FORWARD TIMING (ms) ==="]
    for m in ms:
        m = min(m, N)  # 安全范围
        t  = t_all[:m, :].reshape(-1, 1)            # (m*T, 1)
        d0 = np.repeat(d_all[:m, :1], T, axis=1).reshape(-1, 1)
        w0 = np.repeat(w_all[:m, :1], T, axis=1).reshape(-1, 1)
        x_m = np.concatenate([t, d0, w0], axis=1).astype(np.float32)  # (m*T, 3)

        model.eval()
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            preds = []
            for i in range(0, x_m.shape[0], batch_size):
                xb = torch.from_numpy(x_m[i:i+batch_size]).to(device, non_blocking=True)
                yb = model(xb)
                preds.append(yb.detach().cpu().numpy())

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        ms_val = (t1 - t0) * 1e3
        s = f"[bench] {m:>6d} traj  ->  {ms_val:.3f} ms"
        print(s)
        lines.append(s)

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

def save_one_traj(t, y_true, y_pred, best_idx, N, T, save_path):
    """
    绘制 delta 和 omega 分图的最佳轨迹
    输入 y_true, y_pred 是 (N, T, 2)
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # δ
    axs[0].plot(t, y_true[best_idx, :, 0], label="delta true", color="blue")
    axs[0].plot(t, y_pred[best_idx, :, 0], label="delta pred", color="orange", linestyle="--")
    axs[0].set_ylabel("delta (rad)")
    axs[0].legend()

    # ω
    axs[1].plot(t, y_true[best_idx, :, 1], label="omega true", color="blue")
    axs[1].plot(t, y_pred[best_idx, :, 1], label="omega pred", color="orange", linestyle="--")
    axs[1].set_ylabel("omega (rad/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()

    plt.suptitle(" Exemple prediction vs. ground truth")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==============================
# main
# ==============================
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    device = pick_device(CONFIG["device"])
    print(f"[INFO] device={device}")

    # 1) 构建模型并加载权重
    print(f"[INFO] loading model: {CONFIG['model_path']}")
    model = Network(
        input_size = CONFIG["input_size"],
        hidden_size= CONFIG["hidden_size"],
        output_size= CONFIG["output_size"],
        num_layers = CONFIG["num_layers"],
    ).to(device)
    load_checkpoint_strict(model, CONFIG["model_path"], device)

    # 2) 读数据 & 展平成 (N*T,·)
    print(f"[INFO] loading dataset: {CONFIG['dataset_path']}")
    data = load_dataset(CONFIG["dataset_path"])
    x_np, y_np, N, T = make_xy_from_dataset(data)
    t = x_np[:T, 0]
    total = x_np.shape[0]
    print(f"[INFO] Traj={N} | T={T} | Samples={total}")

    # 3) 纯前向（计时，ms）
    if CONFIG["one_shot"]:
        y_pred, sec = forward_one_shot(model, x_np, device)
    else:
        y_pred, sec = forward_in_batches(model, x_np, device, CONFIG["batch_size"])

    total_ms = sec * 1e3
    per_traj_ms = total_ms / N
    print("\n=== FORWARD (CPU/GPU, ms) ===")
    print(f"[RESULT] Total time     = {total_ms:.3f} ms")
    print(f"[RESULT] Per-trajectory = {per_traj_ms:.6f} ms/traj")

    # 4) 全局指标（不计时）
    mae, mse, maxae = compute_metrics(y_np, y_pred)
    print("\n=== METRICS (NOT TIMED) ===")
    print(f"MAE     = {mae:.6e}")
    print(f"MSE    = {mse:.6e}")
    print(f"Max AE  = {maxae:.6e}")
    # 4b) 保存所有 loss 指标
    t_MAE, t_MSE, t_MaxAE = error_over_time(y_np, y_pred, N, T)
    final_max_mae = float(np.max(t_MaxAE[-1]))  # (T,2) 最后一个时间点，δ/ω 中最大的那个绝对误差

    metrics_txt = os.path.join(CONFIG["out_dir"], "loss_metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write("=== Vanilla 4000: TEST LOSS METRICS ===\n")
        f.write(f"MAE             = {mae:.6e}\n")
        f.write(f"MSE            = {mse:.6e}\n")
        f.write(f"Max AE          = {maxae:.6e}\n")
    print(f"[OK] Saved metrics to: {metrics_txt}")

    # 5) over-time 指标 + PDF（每个指标一页，δ/ω同图）
    # 计算加权平均
    mae_avg_t = 0.5 * (t_MAE[:, 0] + t_MAE[:, 1])
    mse_avg_t = 0.5 * (t_MSE[:, 0] + t_MSE[:, 1])
    maxae_avg_t = 0.5 * (t_MaxAE[:, 0] + t_MaxAE[:, 1])

    # 替代原来 save_two_lines
    save_single_line(
        t, mae_avg_t,
        title="MAE over time (avg)",
        ylabel="MAE",
        out_pdf=os.path.join(CONFIG["out_dir"], "mae_over_time_avg.pdf")
    )
    save_single_line(
        t, mse_avg_t,
        title="MSE over time (avg)",
        ylabel="MSE",
        out_pdf=os.path.join(CONFIG["out_dir"], "mse_over_time_avg.pdf")
    )
    save_single_line(
        t, maxae_avg_t,
        title="Max MAE over time (avg)",
        ylabel="Max MAE",
        out_pdf=os.path.join(CONFIG["out_dir"], "max_mae_over_time_avg.pdf")
    )

    # 6) Boxplot（每条轨迹的 mean MAE）
    err = y_pred.reshape(N, T, 2) - y_np.reshape(N, T, 2)
    mae_all = np.abs(err).mean(axis=1)  # (N, 2) 每条轨迹在 T 个时间点的 mean MAE（δ 和 ω）
    mae_avg = mae_all.mean(axis=1)  # (N,) 每条轨迹的 δ 和 ω 的平均 MAE

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [mae_all[:, 0], mae_all[:, 1], mae_avg],
        tick_labels=["delta", "omega", "avg"],
        showfliers=False,  # 🚫 不显示离群点（小圆圈）
        patch_artist=True,  # 🎨 箱体上色
        boxprops=dict(facecolor='lightgray', color='black'),
        medianprops=dict(color='red', linestyle='-', linewidth=2),  # ✅ 中位数线（红虚线）
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    plt.title("MAE distribution")
    plt.ylabel("MAE")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["out_dir"], "box_final_mae.pdf"))
    plt.close()

    # === 新增：保存仅 "delta + omega 的平均 MAE" ===
    mae_avg = 0.5 * (mae_all[:, 0] + mae_all[:, 1])  # shape: (N,)

    # 1. 存为 npy
    # out_avg_path = os.path.join(CONFIG["out_dir"], "mae_avg_per_traj.npy")
    # np.save(out_avg_path, mae_avg)
    # print(f"[OK] Saved per-trajectory avg MAE to: {out_avg_path}")

    # 2. 画 boxplot 图，只画 avg
    plt.figure(figsize=(4, 4))
    plt.boxplot(
        [mae_avg],
        tick_labels=["avg"],
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray', color='black'),
        medianprops=dict(color='red', linestyle='-', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black')
    )
    plt.title("Avg MAE per trajectory (delta + omega)")
    plt.ylabel("MAE")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["out_dir"], "box_avg_only.pdf"))
    plt.close()
    # 7) 选“最小测试损失”的那条轨迹并画一页
    # best_idx, mae_traj = pick_best_traj_idx(y_np, y_pred, N, T)
    # save_one_traj(t,y_np.reshape(N, T, 2),y_pred.reshape(N, T, 2),best_idx,N, T,os.path.join(CONFIG["out_dir"], "best_traj.pdf"))
    # # np.save(os.path.join(CONFIG["out_dir"], "per_traj_mae.npy"), mae_traj)

    # === 新增：画出前 30 条最小 MAE 的轨迹 ===
    # 计算每条轨迹的 MAE（delta 和 omega 的平均）
    err = y_pred.reshape(N, T, 2) - y_np.reshape(N, T, 2)
    mae_all = np.abs(err).mean(axis=1)  # shape: (N, 2)
    mae_avg = 0.5 * (mae_all[:, 0] + mae_all[:, 1])  # shape: (N,)

    # 找出前 30 条最小 MAE 的轨迹索引
    top_k = 30
    top_k_idx = np.argsort(mae_avg)[:top_k]

    # 创建目录用于存储这些轨迹图
    traj_dir = os.path.join(CONFIG["out_dir"], "top_30_trajs")
    os.makedirs(traj_dir, exist_ok=True)

    # 画图
    for i, idx in enumerate(top_k_idx):
        fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

        # δ
        axs[0].plot(t, y_np.reshape(N, T, 2)[idx,:,0], label="delta true", color="blue")
        axs[0].plot(t, y_pred.reshape(N, T, 2)[idx,:,0], label="delta pred", color="orange", linestyle="--")
        axs[0].set_ylabel("delta (rad)")
        axs[0].legend()

        # ω
        axs[1].plot(t, y_np.reshape(N, T, 2)[idx,:,1], label="omega true", color="blue")
        axs[1].plot(t, y_pred.reshape(N, T, 2)[idx,:,1], label="omega pred", color="orange", linestyle="--")
        axs[1].set_ylabel("omega (rad/s)")
        axs[1].set_xlabel("Time (s)")
        axs[1].legend()

        plt.suptitle(f"Test trajectory and ground truth")
        plt.tight_layout()
        fig.savefig(os.path.join(traj_dir, f"traj_{i+1:02d}_idx{idx}.pdf"))
        plt.close()

    print(f"[OK] Saved top-{top_k} trajectory plots to: {traj_dir}")

    # 8) 论文口径的计时（写文本）
    bench_like_paper(model,data,device,os.path.join(CONFIG["out_dir"], "timing_pinn.txt"),ms=(100, 1000, 100000))
    print(f"\n[OK] PDFs saved to: {CONFIG['out_dir']}")
    # 9) 每 100ms 的 MAE boxplot（合并 delta 和 omega）
    # 获取每个时间点对应的时刻（以秒为单位）
    t_all = x_np[:T, 0]
    duration = t_all[-1] - t_all[0]  # 总时间，比如 1.0s
    interval = 0.05  # 每个时间段长度 100ms
    num_bins = int(duration / interval)

    print(f"[INFO] Making MAE boxplot over {num_bins} bins of {interval:.3f}s")

    # reshape 为 (N,T,2)
    y_true = y_np.reshape(N, T, 2)
    y_pred = y_pred.reshape(N, T, 2)
    abs_err = np.abs(y_true - y_pred)  # (N,T,2)
    mae_per_point = 0.5 * (abs_err[:,:,0] + abs_err[:,:,1])  # (N,T)，delta+omega 的 avg MAE

    # 每个时间段做一个 boxplot
    bin_edges = np.linspace(t_all[0], t_all[-1], num_bins + 1)
    bin_indices = np.digitize(t_all, bin_edges) - 1  # (T,) ∈ [0, num_bins-1]

    # 每段一个 box（10个），每个 box 里是 N 条轨迹在该时间段内的误差
    boxes = [[] for _ in range(num_bins)]
    for i in range(num_bins):
        # 选中这个 bin 里的时间点索引
        idx = np.where(bin_indices == i)[0]
        if len(idx) == 0:
            continue
        # 从所有轨迹中提取这些时刻的误差，平均成一个值
        box_data = mae_per_point[:, idx].mean(axis=1)  # (N,)
        boxes[i] = box_data

    # 画图
    plt.figure(figsize=(12, 4))
    plt.boxplot(boxes, positions=[(i + 0.5) * interval for i in range(num_bins)],
                widths=0.04,
                patch_artist=True,
                boxprops=dict(facecolor='lightgray', color='black'),
                medianprops=dict(color='red', linestyle='-', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                showfliers=False)
    plt.yscale("log")
    plt.title("MAE distribution per 100ms interval")
    # 设置 x 轴刻度：10 个位置，每个位置显示一位小数，比如 0.1, 0.2, ..., 1.0
    tick_positions = [(i + 0.5) * interval for i in range(num_bins)]
    tick_labels = [f"{pos:.1f}" for pos in tick_positions]
    # 自动生成20个tick位置
    positions = [(i + 0.5) * interval for i in range(num_bins)]
    labels = [f"{pos:.2f}" for pos in positions]

    # 只显示一半的tick，防止挤压
    plt.xticks(
        ticks=positions[::2],
        labels=labels[::2]
    )
    plt.title("MAE distribution per 100ms interval")
    plt.xlabel("Time (s)")
    plt.ylabel("Avg MAE (delta + omega)")
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.xlim(0, duration)  # ✅ 精准控制横轴范围，不再留白
    # （可选）x 轴刻度自定义，看需要是否加：
    # plt.xticks(positions[::2], labels[::2])

    plt.tight_layout()
    out_box_path = os.path.join(CONFIG["out_dir"], "box_avg_100ms_intervals.pdf")
    plt.savefig(out_box_path)
    plt.close()
    print(f"[OK] Saved time-boxed MAE to: {out_box_path}")
    return y_true, y_pred, t
import matplotlib.pyplot as plt
import numpy as np

# 假设你有这些数组：
# y_true, y_pred shape: [N, T, 2]，N条轨迹，T个时间点，2个变量（delta, omega）
# time: [T]

def plot_multiple_trajectories(y_true, y_pred, time, start_idx=0, num_per_plot=15, save_path='output.pdf'):
    fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(12, 15))  # 3列×5行
    axs = axs.flatten()

    for i in range(num_per_plot):
        idx = start_idx + i
        ax = axs[i]
        delta_true = y_true[idx, :, 0]
        delta_pred = y_pred[idx, :, 0]
        omega_true = y_true[idx, :, 1]
        omega_pred = y_pred[idx, :, 1]

        ax.plot(time, delta_true, label='delta true', color='blue')
        ax.plot(time, delta_pred, label='delta pred', linestyle='--', color='orange')
        ax.plot(time, omega_true, label='omega true', color='green')
        ax.plot(time, omega_pred, label='omega pred', linestyle='--', color='red')

        ax.set_title(f'Traj {idx}')
        ax.set_xticks([])
        ax.set_yticks([])

        if i == 0:
            ax.legend(fontsize=6)

    # 清除空余子图（如果不足15条）
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
import matplotlib.pyplot as plt

def plot_true_pred_subplots(y_true, y_pred, t, traj_indices, save_path):
    """
    绘制多条轨迹的预测与真实值（delta 和 omega），以 subplot 方式集中展示

    y_true, y_pred: [N, T, 2]，分别为 delta 和 omega
    t: [T]，时间轴
    traj_indices: List[int]，想要画的轨迹索引
    save_path: 保存路径
    """
    num_traj = len(traj_indices)
    fig, axs = plt.subplots(num_traj, 2, figsize=(10, 2.5*num_traj), sharex=True)

    if num_traj == 1:
        axs = axs.reshape(1, 2)

    for i, idx in enumerate(traj_indices):
        axs[i, 0].plot(t, y_true[idx, :, 0], label='delta true', color='blue')
        axs[i, 0].plot(t, y_pred[idx, :, 0], label='delta pred', color='orange', linestyle='--')
        axs[i, 0].set_ylabel(f"Traj {idx}")
        if i == 0:
            axs[i, 0].set_title("δ (rad)")

        axs[i, 1].plot(t, y_true[idx, :, 1], label='omega true', color='blue')
        axs[i, 1].plot(t, y_pred[idx, :, 1], label='omega pred', color='orange', linestyle='--')
        if i == 0:
            axs[i, 1].set_title("ω (rad/s)")

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def save_single_traj_plot(y_true, y_pred, time, idx, save_dir="."):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))  # δ 和 ω 各一张图

    labels = ["δ", "ω"]
    for i in range(2):
        axs[i].plot(time, y_true[idx, :, i], label="True", color='blue')
        axs[i].plot(time, y_pred[idx, :, i], label="Pred", color='orange', linestyle='--')
        axs[i].set_title(f"{labels[i]} (Trajectory {idx})")
        axs[i].legend()
        axs[i].set_xlabel("Time (s)")

    plt.tight_layout()
    path = f"{save_dir}/traj_{idx}_true_vs_pred.pdf"
    fig.savefig(path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Saved traj {idx} comparison to: {path}")
if __name__ == "__main__":
    y_true, y_pred, t = main()
    traj_ids_to_plot = list(range(15))   # 可换成 15–30 等
    save_path = os.path.join(CONFIG["out_dir"], "pred_vs_true_15trajs.pdf")
    plot_true_pred_subplots(y_true, y_pred, t, traj_ids_to_plot, save_path)    # ✅ 清理缓存，防止内存累积导致崩溃
    save_single_traj_plot(y_true, y_pred, t, idx=10, save_dir=".")

    import gc
    # for var in ["model", "x_np", "y_pred"]:
    #     if var in locals():
    #         del globals()[var]
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[INFO] GPU cache cleared.")
    else:
        print("[INFO] CPU mode – no CUDA cache to clear.")
