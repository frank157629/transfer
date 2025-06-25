from omegaconf import OmegaConf
from src.nn.nn_dataset import DataSampler      # 你的类

# 1) 载入同一个 yaml
cfg = OmegaConf.load("src/conf/setup_dataset_nn.yaml")

# 2) 实例化 DataSampler（会自动完成切分）
sampler = DataSampler(cfg)

print("=== 全集 ===")
print("x:", sampler.x.shape, "y:", sampler.y.shape)

print("\n=== 切分结果 ===")
print("train :", sampler.x_train.shape,
      f"→ 轨迹数 {sampler.x_train.shape[0] // sampler.sample_per_traj}")
print("val   :", sampler.x_val.shape,
      f"→ 轨迹数 {sampler.x_val.shape[0] // sampler.sample_per_traj}")
print("test  :", sampler.x_test.shape,
      f"→ 轨迹数 {sampler.x_test.shape[0] // sampler.sample_per_traj}")

# 3) 想直观看看某几条 val/test 轨迹
import matplotlib.pyplot as plt
idx_val = 0                                    # 第 1 条验证轨迹
n_pt     = int(sampler.sample_per_traj)        # 一条轨迹有多少点

# 取出时间与一个状态量（比如 gamma）来画
t_val  = sampler.x_val[idx_val*n_pt:(idx_val+1)*n_pt, 0]
g_val  = sampler.y_val[idx_val*n_pt:(idx_val+1)*n_pt, 0]

plt.plot(t_val, g_val)
plt.title(f"val traj {idx_val} - gamma")
plt.xlabel("t"); plt.ylabel("gamma")
plt.show()