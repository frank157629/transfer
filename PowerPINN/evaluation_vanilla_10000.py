"PowerPINN/evaluation_vanilla_10000.py"
"""
Vanilla model evaluation on a dataset shaped (N, 3, T) with channels [t, delta, omega].

What this script does
---------------------
1) Loads a trained "vanilla" network and a test dataset of shape (N,3,T).
   The network input is [t, delta0, omega0] and it predicts [delta(t), omega(t)].

2) Runs batched inference (or one-shot), reports timing, and computes:
   - Global metrics: MAE, MSE, MaxAE (over all trajectories and timesteps)
   - Time-resolved metrics: per-timestep MAE/MSE/MaxAE (δ and ω separately)

3) Saves:
   - Time-series PDFs: mae_over_time_delta/omega, mse_over_time_delta/omega,
     maxae_over_time_delta/omega
   - One example trajectory plot: example_traj.pdf
   - Time-binned (50 ms) MAE boxplots for δ and ω
   - NPY arrays for time-series metrics and per-trajectory avg MAE
   - A paper-style timing report

Inputs
------
- model_path: checkpoint .pth (either raw state_dict or {'model_state_dict': ...})
- dataset_path: pickled test set of shape (N,3,T) with [t, delta, omega]

Outputs
-------
- PDFs: mae_over_time_*.pdf, mse_over_time_*.pdf, maxae_over_time_*.pdf,
        example_traj.pdf, box_50ms_delta.pdf, box_50ms_omega.pdf
- NPYs: mae_over_time_*.npy, mse_over_time_*.npy, maxae_over_time_*.npy,
        mae_avg_per_traj.npy
- TXT: timing_vanilla.txt, loss_metrics_separate.txt

Config knobs
------------
- device: "auto" | "cpu" | "cuda"
- one_shot / batch_size: inference mode
- network sizes must match training
"""

import os, time, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ==============================
# Configuration (edit paths)
# ==============================
CONFIG = {
    "model_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/vanilla/vanilla_model_10000/GFL_2nd_orderDynamicNN_1_750_8000000_1000_1000000_None_None_1_0_0_0_Static_20250814-101254.pth",
    "dataset_path": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/lhs_sampling/dataset_v8_mixed_k1000.pkl",
    "out_dir": "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/vanilla/reports_vanilla_10000",
    "device": "auto",

    # Must match training
    "input_size": 3,      # [t, delta0, omega0]
    "hidden_size": 128,
    "output_size": 2,     # [delta(t), omega(t)]
    "num_layers": 4,

    # Inference mode
    "one_shot": False,        # set True for small datasets
    "batch_size": 131072,     # along (N*T)
}


# ==============================
# Model (must match training)
# ==============================
class Network(nn.Module):
    """
    Input : [t, delta0, omega0]  (dim=3)
    Output: [delta(t), omega(t)] (dim=2)
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
# Utilities
# ==============================
def pick_device(name: str) -> torch.device:
    """Select device by name or auto-detect CUDA."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def load_checkpoint_strict(model: nn.Module, path: str, device: torch.device):
    """Load a state dict; supports raw state_dict or dict with 'model_state_dict' key."""
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print("[INFO] model loaded (strict=True).")

def load_dataset(path: str) -> np.ndarray:
    """Load dataset of shape (N,3,T) with channels [t, delta, omega]."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = np.array(data, dtype=np.float32)
    assert data.ndim == 3 and data.shape[1] == 3, f"expect (N,3,T), got {data.shape}"
    return data

def make_xy_from_dataset(data: np.ndarray):
    """
    Build supervised pairs from dataset.

    Args:
        data: np.ndarray of shape (N,3,T) with [t, delta(t), omega(t)]

    Returns:
        x_np: (N*T, 3)  -> inputs  [t, delta0, omega0]
        y_np: (N*T, 2)  -> targets [delta(t), omega(t)]
        N:    int       -> number of trajectories
        T:    int       -> points per trajectory
    """
    N, _, T = data.shape
    t_all = data[:, 0, :]
    d_all = data[:, 1, :]
    w_all = data[:, 2, :]

    # Use the first sample of each trajectory as IC: (delta0, omega0)
    d0 = d_all[:, 0][:, None]
    w0 = w_all[:, 0][:, None]

    # Repeat IC across all time steps in the trajectory
    d0_rep = np.repeat(d0, T, axis=1)
    w0_rep = np.repeat(w0, T, axis=1)

    x_np = np.stack([t_all, d0_rep, w0_rep], axis=2).reshape(-1, 3).astype(np.float32)
    y_np = np.stack([d_all,  w_all ],        axis=2).reshape(-1, 2).astype(np.float32)
    return x_np, y_np, N, T

@torch.no_grad()
def forward_one_shot(model: nn.Module, x_np: np.ndarray, device: torch.device):
    """Single forward pass with all (N*T) inputs at once. Returns predictions and wall time."""
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
    """Batched forward for large inputs; concatenates predictions and measures wall time."""
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
    """Global scalar metrics: MAE / MSE / Max Absolute Error."""
    err  = y_pred - y_true
    mae  = float(np.mean(np.abs(err)))
    mse  = float(np.mean(err**2))
    maxe = float(np.max(np.abs(err)))
    return mae, mse, maxe

def error_over_time(y_true_flat, y_pred_flat, N, T):
    """
    Per-time metrics across all trajectories.

    Returns:
        t_MAE  : (T,2) mean |error| over trajectories, per time step
        t_MSE  : (T,2) mean  error^2 over trajectories, per time step
        t_MaxAE: (T,2) max  |error| over trajectories, per time step
    """
    err = (y_pred_flat - y_true_flat).reshape(N, T, 2)
    t_MAE   = np.mean(np.abs(err), axis=0)
    t_MSE   = np.mean(err**2,      axis=0)
    t_MaxAE = np.max(np.abs(err),  axis=0)
    return t_MAE, t_MSE, t_MaxAE

def save_single_line(t, y, title, ylabel, out_pdf):
    """Plot a single time series y(t)."""
    plt.figure(figsize=(7, 4))
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(ls="--", alpha=.3)
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

def save_example_traj(t, y_true, y_pred, save_path):
    """
    Plot one random trajectory as two subplots (delta, omega).
    y_true, y_pred: (N,T,2)
    """
    N, T, _ = y_true.shape
    idx = np.random.randint(0, N)

    fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # δ
    axs[0].plot(t, y_true[idx, :, 0], label="delta true")
    axs[0].plot(t, y_pred[idx, :, 0], "--", label="delta pred")
    axs[0].set_ylabel("delta (rad)")
    axs[0].legend()

    # ω
    axs[1].plot(t, y_true[idx, :, 1], label="omega true")
    axs[1].plot(t, y_pred[idx, :, 1], "--", label="omega pred")
    axs[1].set_ylabel("omega (rad/s)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()

    plt.suptitle("Example prediction vs. ground truth")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def bench_like_paper(model, data_np, device, out_txt, ms=(1, 50, 500), batch_size=131072):
    """
    Measure pure forward-pass wall time (ms) for m trajectories (no IO).
    Writes a short, paper-style report to out_txt.
    """
    N, _, T = data_np.shape
    t_all = data_np[:, 0, :]
    d_all = data_np[:, 1, :]
    w_all = data_np[:, 2, :]

    lines = ["=== PURE FORWARD TIMING (ms) ==="]
    for m in ms:
        m = min(m, N)
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


# ==============================
# main
# ==============================
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    device = pick_device(CONFIG["device"])
    print(f"[INFO] device={device}")

    # 1) Build model & load weights
    print(f"[INFO] loading model: {CONFIG['model_path']}")
    model = Network(
        input_size = CONFIG["input_size"],
        hidden_size= CONFIG["hidden_size"],
        output_size= CONFIG["output_size"],
        num_layers = CONFIG["num_layers"],
    ).to(device)
    load_checkpoint_strict(model, CONFIG["model_path"], device)

    # 2) Load dataset & flatten to (N*T, ·)
    print(f"[INFO] loading dataset: {CONFIG['dataset_path']}")
    data = load_dataset(CONFIG["dataset_path"])
    x_np, y_np, N, T = make_xy_from_dataset(data)
    t = x_np[:T, 0]  # common time axis
    total = x_np.shape[0]
    print(f"[INFO] Traj={N} | T={T} | Samples={total}")

    # 3) Forward timing (ms)
    if CONFIG["one_shot"]:
        y_pred, sec = forward_one_shot(model, x_np, device)
    else:
        y_pred, sec = forward_in_batches(model, x_np, device, CONFIG["batch_size"])
    total_ms = sec * 1e3
    per_traj_ms = total_ms / N
    print("\n=== FORWARD (CPU/GPU, ms) ===")
    print(f"[RESULT] Total time     = {total_ms:.3f} ms")
    print(f"[RESULT] Per-trajectory = {per_traj_ms:.6f} ms/traj")

    # 4) Global metrics (not timed)
    mae, mse, maxae = compute_metrics(y_np, y_pred)
    print("\n=== METRICS (NOT TIMED) ===")
    print(f"MAE     = {mae:.6e}")
    print(f"MSE     = {mse:.6e}")
    print(f"Max AE  = {maxae:.6e}")

    # 4b) Time-resolved metrics (δ/ω separated)
    t_MAE, t_MSE, t_MaxAE = error_over_time(y_np, y_pred, N, T)

    # Final-time split (for a compact log)
    final_mae_delta   = t_MAE[-1, 0]
    final_mae_omega   = t_MAE[-1, 1]
    final_mse_delta   = t_MSE[-1, 0]
    final_mse_omega   = t_MSE[-1, 1]
    final_maxae_delta = t_MaxAE[-1, 0]
    final_maxae_omega = t_MaxAE[-1, 1]

    print("Test loss vanilla (separated at final time):")
    print(f"MAE (delta)    = {final_mae_delta:.6e}")
    print(f"MAE (omega)    = {final_mae_omega:.6e}")
    print(f"MSE (delta)    = {final_mse_delta:.6e}")
    print(f"MSE (omega)    = {final_mse_omega:.6e}")
    print(f"MaxAE (delta)  = {final_maxae_delta:.6e}")
    print(f"MaxAE (omega)  = {final_maxae_omega:.6e}")

    with open(os.path.join(CONFIG["out_dir"], "loss_metrics_separate.txt"), "a") as f:
        f.write("\nTest loss vanilla (separated at final time):\n")
        f.write(f"MAE (delta)    = {final_mae_delta:.6e}\n")
        f.write(f"MAE (omega)    = {final_mae_omega:.6e}\n")
        f.write(f"MSE (delta)    = {final_mse_delta:.6e}\n")
        f.write(f"MSE (omega)    = {final_mse_omega:.6e}\n")
        f.write(f"MaxAE (delta)  = {final_maxae_delta:.6e}\n")
        f.write(f"MaxAE (omega)  = {final_maxae_omega:.6e}\n")

    # 5) Over-time plots (δ/ω separated)
    save_single_line(t, t_MAE[:,0], "MAE over time (delta)", "MAE",
                     os.path.join(CONFIG["out_dir"], "mae_over_time_delta.pdf"))
    save_single_line(t, t_MAE[:,1], "MAE over time (omega)", "MAE",
                     os.path.join(CONFIG["out_dir"], "mae_over_time_omega.pdf"))

    save_single_line(t, t_MSE[:,0], "MSE over time (delta)", "MSE",
                     os.path.join(CONFIG["out_dir"], "mse_over_time_delta.pdf"))
    save_single_line(t, t_MSE[:,1], "MSE over time (omega)", "MSE",
                     os.path.join(CONFIG["out_dir"], "mse_over_time_omega.pdf"))

    save_single_line(t, t_MaxAE[:,0], "MaxAE over time (delta)", "MaxAE",
                     os.path.join(CONFIG["out_dir"], "maxae_over_time_delta.pdf"))
    save_single_line(t, t_MaxAE[:,1], "MaxAE over time (omega)", "MaxAE",
                     os.path.join(CONFIG["out_dir"], "maxae_over_time_omega.pdf"))

    # 6) Per-trajectory mean MAE (save and quick boxplot of averages if you want)
    err = y_pred.reshape(N, T, 2) - y_np.reshape(N, T, 2)
    mae_all = np.abs(err).mean(axis=1)       # (N,2): mean over time for each traj, per variable
    mae_avg = mae_all.mean(axis=1)           # (N,) : average over variables per trajectory
    np.save(os.path.join(CONFIG["out_dir"], "mae_avg_per_traj.npy"), mae_avg)
    print(f"[OK] Saved per-trajectory avg MAE to: {os.path.join(CONFIG['out_dir'], 'mae_avg_per_traj.npy')}")

    # 7) Example trajectory (random)
    save_example_traj(
        t,
        y_np.reshape(N, T, 2),
        y_pred.reshape(N, T, 2),
        os.path.join(CONFIG["out_dir"], "example_traj.pdf")
    )

    # 8) Timing report
    bench_like_paper(model, data, device,
                     os.path.join(CONFIG["out_dir"], "timing_vanilla.txt"),
                     ms=(1, 50, 500))

    # 9) MAE boxplots per fixed time interval (delta & omega separately)
    #    Using 0.05 s bins (≈ 50 ms for a 1 s trajectory)
    t_all = x_np[:T, 0]
    duration = t_all[-1] - t_all[0]
    interval = 0.05
    num_bins = int(duration / interval)
    print(f"[INFO] Making MAE boxplot over {num_bins} bins of {interval:.3f}s")

    y_true = y_np.reshape(N, T, 2)
    y_hat  = y_pred.reshape(N, T, 2)
    abs_err = np.abs(y_true - y_hat)            # (N,T,2)
    mae_d   = abs_err[:, :, 0]                  # (N,T)
    mae_w   = abs_err[:, :, 1]                  # (N,T)

    bin_edges = np.linspace(t_all[0], t_all[-1], num_bins + 1)
    bin_indices = np.digitize(t_all, bin_edges) - 1  # (T,) in [0, num_bins-1]

    # delta
    boxes_d = [[] for _ in range(num_bins)]
    for i in range(num_bins):
        idx = np.where(bin_indices == i)[0]
        if len(idx) == 0: continue
        boxes_d[i] = mae_d[:, idx].mean(axis=1)   # (N,)
    pos = [(i + 0.5) * interval for i in range(num_bins)]
    lab = [f"{p:.2f}" for p in pos]
    plt.figure(figsize=(12, 4))
    plt.boxplot(
        boxes_d, positions=pos, widths=0.04,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray', color='black'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        showfliers=False
    )
    plt.yscale("log")
    plt.title("MAE distribution per 50 ms interval (delta)")
    plt.xlabel("Time (s)")
    plt.ylabel("MAE (delta)")
    plt.xticks(pos[::2], lab[::2])
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["out_dir"], "box_50ms_delta.pdf"))
    plt.close()

    # omega
    boxes_w = [[] for _ in range(num_bins)]
    for i in range(num_bins):
        idx = np.where(bin_indices == i)[0]
        if len(idx) == 0: continue
        boxes_w[i] = mae_w[:, idx].mean(axis=1)
    plt.figure(figsize=(12, 4))
    plt.boxplot(
        boxes_w, positions=pos, widths=0.04,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray', color='black'),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        showfliers=False
    )
    plt.yscale("log")
    plt.title("MAE distribution per 50 ms interval (omega)")
    plt.xlabel("Time (s)")
    plt.ylabel("MAE (omega)")
    plt.xticks(pos[::2], lab[::2])
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["out_dir"], "box_50ms_omega.pdf"))
    plt.close()

    # 10) Save time-series arrays (for reproducible plotting elsewhere)
    np.save(os.path.join(CONFIG["out_dir"], "mae_over_time_delta.npy"), t_MAE[:, 0])
    np.save(os.path.join(CONFIG["out_dir"], "mae_over_time_omega.npy"), t_MAE[:, 1])
    np.save(os.path.join(CONFIG["out_dir"], "mse_over_time_delta.npy"), t_MSE[:, 0])
    np.save(os.path.join(CONFIG["out_dir"], "mse_over_time_omega.npy"), t_MSE[:, 1])
    np.save(os.path.join(CONFIG["out_dir"], "maxae_over_time_delta.npy"), t_MaxAE[:, 0])
    np.save(os.path.join(CONFIG["out_dir"], "maxae_over_time_omega.npy"), t_MaxAE[:, 1])
    print("[OK] Saved all time-series arrays as .npy.")

    print(f"\n[OK] Reports saved to: {CONFIG['out_dir']}")
    return y_true, y_hat, t, t_MAE, t_MSE, t_MaxAE


if __name__ == "__main__":
    y_true, y_pred, t, t_MAE, t_MSE, t_MaxAE = main()

    # Optional: plot any specific trajectories again if needed
    # (Kept minimal here to mirror the PINN script layout)
    # Clean up GPU/CPU caches for long sessions
    import gc
    to_del = ["y_pred"]  # extend as needed
    for name in to_del:
        if name in globals():
            del globals()[name]
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[INFO] GPU cache cleared.")
    else:
        print("[INFO] CPU mode – no CUDA cache to clear.")