import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.nn.nn_dataset import DataSampler
from src.nn.nn_model import Net, Network, PinnA, FullyConnectedResNet, Kalm
from src.functions import *
from src.nn.early_stopping import EarlyStopping
from src.ode.sm_models_d import SynchronousMachineModels
from src.ode.gfl_models_d import GridFollowingConverterModels
import wandb
import torch.autograd.functional as func
from src.nn.gradient_based_weighting import PINNWeighting
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from src.ode.gfl_models_d import calculate_frequency

""" (1)Here is the class for the Vanilla training. 
    However the Speed was very slow and I went to use 
    pinn_actions. 
    (1)Setting three other weights 
    to zero in "setup_dataset_pinn_gfl.yaml" was a reliable approach.
    the model is only relying on the dataloss. There was nowhere 
    found in the code where the three other losses are still affecting 
    the losses.
"""
#Haitian, new for Vanilla NN training
class VanillaNeuralNetworkActions():
    def __init__(self, cfg):
        self.cfg = cfg
        set_random_seeds(cfg.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_loader = DataSampler(cfg)
        self.input_dim = self.data_loader.input_dim
        self.output_dim = self.input_dim - 1
        self.model_flag = cfg.model.model_flag
        self.model = self.define_nn_model()
        self.weight_init(self.model, cfg.network.weight_init)
        self.criterion = self.custom_loss(cfg.network.loss_criterion)
        self.criterion_mae = nn.L1Loss()
        self.optimizer = self.custom_optimizer(cfg.network.optimizer, cfg.network.lr)
        self.scheduler = self.custom_learning_rate(cfg.network.lr_scheduler)
        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=cfg.network.early_stopping_patience,
                                            verbose=True,
                                            delta=cfg.network.early_stopping_min_delta)
        self.sample_per_traj = int(self.data_loader.sample_per_traj)

    def log_plot(self, output, target, epoch, run, x_test, type="val", starting_traj=0, total_traj=1):
        import numpy as np
        import matplotlib.pyplot as plt
        import wandb
        from src.ode.gfl_models_d import calculate_frequency

        pts_per_traj = int(self.data_loader.sample_per_traj)
        max_traj = len(output) // pts_per_traj
        total_traj = min(total_traj, max_traj)
        blk_idx = list(range(starting_traj, starting_traj + total_traj))

        fig, axes = plt.subplots(total_traj, 3, figsize=(18, 3 * total_traj), sharex='col')

        # 如果只有一个轨迹，axes 会变成一维
        if total_traj == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, k in enumerate(blk_idx):
            lo, hi = k * pts_per_traj, (k + 1) * pts_per_traj
            t = x_test[lo:hi, 0].detach().cpu().numpy()
            delta_true = target[lo:hi, 0].detach().cpu().numpy()
            omega_true = target[lo:hi, 1].detach().cpu().numpy()
            delta_pred = output[lo:hi, 0].detach().cpu().numpy()
            omega_pred = output[lo:hi, 1].detach().cpu().numpy()

            f_true = calculate_frequency(omega_true, np.pi * 100)
            f_pred = calculate_frequency(omega_pred, np.pi * 100)

            axd, axw, axf = axes[r, 0], axes[r, 1], axes[r, 2]
            axd.plot(t, delta_true, label="True", lw=1.2)
            axd.plot(t, delta_pred, label="Pred", lw=1.2, ls='--')
            axd.set_title("δ")
            axd.grid(ls='--', alpha=.3)

            axw.plot(t, omega_true, lw=1.2)
            axw.plot(t, omega_pred, lw=1.2, ls='--')
            axw.set_title("ω")
            axw.grid(ls='--', alpha=.3)

            axf.plot(t, f_true, lw=1.2)
            axf.plot(t, f_pred, lw=1.2, ls='--')
            axf.set_title("f = ω / (2π)")
            axf.grid(ls='--', alpha=.3)

            axd.text(-0.05, 0.5, f"traj {k + 1}", transform=axd.transAxes, va='center', ha='right', fontsize=9)

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        axes[-1, 2].set_xlabel("Time (s)")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout()
        gname = f"{type}_epoch{epoch}_traj{blk_idx[0] + 1}-{blk_idx[-1] + 1}"
        run.log({f"Traj_{gname}": wandb.Image(fig)}, commit=False)

        plt.close(fig)
        run.log({}, commit=True)

    def define_nn_model(self):
        print("Selected model: ", self.cfg.network.type)
        if self.cfg.network.type == "StaticNN":
            return Net(self.input_dim, self.cfg.network.hidden_dim, self.output_dim)
        elif self.cfg.network.type == "DynamicNN":
            return Network(self.input_dim, self.cfg.network.hidden_dim,
                           self.output_dim, self.cfg.network.hidden_layers)
        elif self.cfg.network.type == "ResNet":
            return FullyConnectedResNet(self.input_dim, self.cfg.network.hidden_dim,
                                        self.output_dim, num_blocks=2, num_layers_per_block=2)
        elif self.cfg.network.type == "KAN":
            return Kalm(self.input_dim, self.cfg.network.hidden_dim,
                        self.output_dim, self.cfg.network.hidden_layers)
        else:
            raise Exception("Invalid model type.")

    def custom_loss(self, loss_name):
        if loss_name == 'MSELoss':
            return nn.MSELoss()
        elif loss_name == 'L1Loss':
            return nn.L1Loss()
        elif loss_name == 'SmoothL1Loss':
            return nn.SmoothL1Loss()
        else:
            raise Exception("Invalid loss function.")

    def custom_optimizer(self, optimizer_name, learning_rate):
        if optimizer_name == 'Adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adam_decay':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        elif optimizer_name == 'SGD':
            return optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception("Invalid optimizer.")

    def custom_learning_rate(self, lr_name):
        if lr_name == 'StepLR':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif lr_name == 'ExponentialLR':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif lr_name == 'No_scheduler':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        else:
            raise Exception("Invalid LR scheduler.")

    def weight_init(self, module, init_name):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                if init_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                elif init_name == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)

    def forward_pass(self, x):
        return self.model(x)

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            return self.forward_pass(x_test)

    def vanilla_train(self, wandb_run=None):
        x_train, y_train, x_val, y_val = self.data_loader.define_train_val_data_supervised()
        self.x_val, self.y_val = x_val, y_val  # 保存为成员变量

        batch_size = self.cfg.network.batch_size if self.cfg.network.batch_size != "None" else len(x_train)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)

        for epoch in range(self.cfg.network.num_epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                y_pred = self.forward_pass(xb)
                loss = self.criterion(y_pred, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            # Validation（每个 epoch 都评估 val_loss 和 MAE）
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.forward_pass(x_val)
                val_loss = self.criterion(y_val_pred, y_val).item()
                val_mae = self.criterion_mae(y_val_pred, y_val).item()

            # 每 50 个 epoch 输出终端信息 + 画图上传
            if (epoch + 1) % 50 == 0:
                print(f"[{epoch + 1}] Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f} | MAE: {val_mae:.6f}")
                if wandb_run:
                    self.log_plot(
                        output=y_val_pred,
                        target=y_val,
                        epoch=epoch,
                        run=wandb_run,
                        x_test=x_val,
                        type=f"val_epoch_{epoch + 1}",
                        starting_traj=0,
                        total_traj=1  # 你可以设成更大看看更多轨迹
                    )

            # log loss/MAE to wandb 每个 epoch 都上传
            if wandb_run:
                wandb_run.log({
                    "Train_loss": loss.item(),
                    "Val_loss": val_loss,
                    "Val_MAE": val_mae,
                    "epoch": epoch
                })

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        self.save_model("vanilla_final.pth")
        self.test_model(run=wandb_run)

    def test_model(self, starting_traj=0, total_traj=1, run=None):
        total_traj = min(total_traj, self.data_loader.total_test_trajectories)
        x_test, y_test = self.data_loader.define_test_data(starting_traj, self.sample_per_traj, total_traj)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.forward_pass(x_test)
        test_loss = self.criterion(y_pred, y_test)
        mae = self.criterion_mae(y_pred, y_test)
        print(f"Test MSE: {test_loss.item():.6f}, Test MAE: {mae.item():.6f}")
        if run:
            run.log({"Test_loss": test_loss.item(), "Test_MAE": mae.item()})

            # === OVERTIME 计算 ===
            pts_per_traj = self.sample_per_traj
            y_true_np = y_test.detach().cpu().numpy().reshape(-1, pts_per_traj, self.output_dim)
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1, pts_per_traj, self.output_dim)

            mae_overtime = np.mean(np.abs(y_pred_np - y_true_np), axis=(0, 2))  # shape: (T,)
            max_mae_overtime = np.max(np.abs(y_pred_np - y_true_np), axis=(0, 2))  # shape: (T,)
            final_mae = np.mean(np.abs(y_pred_np - y_true_np))  # scalar
            final_mse = np.mean((y_pred_np - y_true_np) ** 2)  # scalar
            final_max_mae = np.max(np.abs(y_pred_np - y_true_np))  # scalar

            run.log({
                "MAE_over_time": wandb.plot.line_series(
                    xs=np.linspace(0, 1, pts_per_traj).tolist(),
                    ys=[mae_overtime.tolist()],
                    keys=["MAE"],
                    title="MAE over time",
                    xname="Time (s)"
                ),
                "Max_MAE_over_time": wandb.plot.line_series(
                    xs=np.linspace(0, 1, pts_per_traj).tolist(),
                    ys=[max_mae_overtime.tolist()],
                    keys=["Max MAE"],
                    title="Max MAE over time",
                    xname="Time (s)"
                ),
                "final_MAE": final_mae,
                "final_MSE": final_mse,
                "final_Max_MAE": final_max_mae
            })

    def save_model(self, name):
        model_path = os.path.join(self.cfg.dirs.model_dir, name)
        torch.save({"model_state_dict": self.model.state_dict()}, model_path)
        print("Model saved:", model_path)

    def load_model(self, name):
        model_path = os.path.join(self.cfg.dirs.model_dir, name)
        model_data = torch.load(model_path)
        self.model.load_state_dict(model_data["model_state_dict"])