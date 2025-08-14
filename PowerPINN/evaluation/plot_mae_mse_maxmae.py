import numpy as np
import matplotlib.pyplot as plt
import os

def load_metrics(folder):
    return {
        "mae": np.load(os.path.join(folder, "mae_avg_t.npy")),
        "mse": np.load(os.path.join(folder, "mse_avg_t.npy")),
        "maxae": np.load(os.path.join(folder, "maxae_avg_t.npy")),
        "time": np.load(os.path.join(folder, "time_axis.npy")),
    }

def plot_compare(metric_name, time, curve1, curve2, label1, label2, save_path):
    plt.figure(figsize=(7, 4))
    plt.plot(time, curve1, label=f"{metric_name} – {label1}", color="blue")
    plt.plot(time, curve2, label=f"{metric_name} – {label2}", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Time")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[OK] Saved: {save_path}")

def main():
    # 替换成你实际的输出文件夹
    pinn_dir = "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/pinn/reports_pinn"
    vanilla_dir = "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation/vanilla/reports_vanilla_10000"

    pinn = load_metrics(pinn_dir)
    vanilla = load_metrics(vanilla_dir)

    assert np.allclose(pinn["time"], vanilla["time"]), "Time mismatch!"

    out_dir = "/Users/nbhsbgnb/PycharmProjects/PythonProject/PowerPINN/evaluation"
    os.makedirs(out_dir, exist_ok=True)

    plot_compare("MAE", pinn["time"], pinn["mae"], vanilla["mae"], "PINN", "Vanilla",
                 os.path.join(out_dir, "compare_mae.pdf"))

    plot_compare("MSE", pinn["time"], pinn["mse"], vanilla["mse"], "PINN", "Vanilla",
                 os.path.join(out_dir, "compare_mse.pdf"))

    plot_compare("Max MAE", pinn["time"], pinn["maxae"], vanilla["maxae"], "PINN", "Vanilla",
                 os.path.join(out_dir, "compare_maxmae.pdf"))

if __name__ == "__main__":
    main()