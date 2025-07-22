import numpy as np
from scipy.integrate import solve_ivp
from src.functions import set_time
import torch
import os
from omegaconf import OmegaConf
import random


class GFM:
    def __init__(self, config):
        self.params_dir = config.dirs.params_dir
        self.machine_num = config.model.machine_num
        self.model_flag = config.model.model_flag
        self.define_machine_params()  # define parameters of GFM
        self.define_system_params()  # define parameters of grid

        # Aggiungo la definizione del device qui:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Definizione dei ratio
        ratios = [
            (0.90, 0.10),
            (0.45, 0.55),
            (0.55, 0.45),
            (0.65, 0.35),
            (0.75, 0.25),
            (0.05, 0.95),
            (0.75, 0.25),
            (0.15, 0.85),
            (0.3, 0.7),
            (0.99, 0.01)
        ]

        # Scelta casuale del ratio
        p_ratio, q_ratio = random.choice(ratios)

        # Impostazione delle potenze di riferimento
        self.p_ref = p_ratio * (1000e3 / self.S_n)
        self.q_ref = q_ratio * (1000e3 / self.S_n)
        self.v_ref = 1.0

    def define_machine_params(self):
        machine_params_path = os.path.join(self.params_dir, "GFM.yaml")
        machine_params = OmegaConf.load(machine_params_path)
        for param in ['lf', 'rf', 'cf', 'kp_v', 'ki_v', 'kp_c', 'ki_c', 'kFFi', 'kFFv',
                      'rg', 'lg', 'rt', 'lt', 's_pf', 's_qv', 'wq']:
            setattr(self, param, getattr(machine_params, param))

    def define_system_params(self):
        system_params_path = os.path.join(self.params_dir, "system_bus.yaml")
        system_params = OmegaConf.load(system_params_path)
        for param in ['V_n_phph', 'S_n', 'f_n', 'omega_B']:
            setattr(self, param, getattr(system_params, param))
        self.w_n = self.omega_B  # Definisci self.w_n come omega_B

    def calculate_voltages(self, v_ref, gamma_q):
        v_ref_d = v_ref + gamma_q
        v_ref_q = 0
        return v_ref_d, v_ref_q

    def calculate_currents(self, vfd, vfq, xi_d, xi_q, v_ref_d, v_ref_q, omega_gfm):
        i_ref_d = self.kp_v * (v_ref_d - vfd) + self.ki_c * xi_d - omega_gfm * self.cf * vfq
        i_ref_q = self.kp_v * (v_ref_q - vfq) + self.ki_c * xi_q + omega_gfm * self.cf * vfd
        return i_ref_d, i_ref_q

    def calculate_grid_voltages(self, theta_grid, theta_gfm, itd, itq, omega_gfm):
        if isinstance(theta_grid, torch.Tensor):
            vgd_inf = 1.0 * torch.cos(theta_grid - theta_gfm)
            vgq_inf = 1.0 * torch.sin(theta_grid - theta_gfm)
        else:
            vgd_inf = 1.0 * np.cos(theta_grid - theta_gfm)
            vgq_inf = 1.0 * np.sin(theta_grid - theta_gfm)

        vgd = vgd_inf - self.rg * itd - self.lg * omega_gfm * itq
        vgq = vgq_inf - self.rg * itq + self.lg * omega_gfm * itd

        return vgd, vgq

    def odequations(self, t, x):
        if self.model_flag == "GFM":
            xi_d, xi_q, vfd, vfq, ifd, ifq, itd, itq, sigma_d, sigma_q, gamma_q, theta_gfm, theta_grid = x

            # Calcola potenze attiva e reattiva istantanee
            p_inst = vfd * itd + vfq * itq
            q_inst = vfq * itd - vfd * itq

            # Calcola frequenza GFM
            omega_gfm = 1.0 + self.s_pf * (self.p_ref - p_inst)

            # Calcola tensioni e correnti di riferimento
            v_ref_d, v_ref_q = self.calculate_voltages(self.v_ref, gamma_q)
            i_ref_d, i_ref_q = self.calculate_currents(vfd, vfq, xi_d, xi_q, v_ref_d, v_ref_q, omega_gfm)
            vgd, vgq = self.calculate_grid_voltages(theta_grid, theta_gfm, itd, itq, omega_gfm)

            # Equazioni ausiliarie
            vmd = self.kp_c * (
                    i_ref_d - ifd + self.kFFi * itd) + self.ki_c * sigma_d - omega_gfm * self.lf * ifq + vfd * self.kFFv
            vmq = self.kp_c * (
                    i_ref_q - ifq + self.kFFi * itq) + self.ki_c * sigma_q + omega_gfm * self.lf * ifd + vfq * self.kFFv
            dxi_d_dt = v_ref_d - vfd
            dxi_q_dt = v_ref_q - vfq
            dvfd_dt = self.w_n * ((ifd - itd) / self.cf + omega_gfm * vfq)
            dvfq_dt = self.w_n * ((ifq - itq) / self.cf - omega_gfm * vfd)
            difd_dt = self.w_n * ((vmd - vfd - self.rf * ifd) / self.lf + omega_gfm * ifq)
            difq_dt = self.w_n * ((vmq - vfq - self.rf * ifq) / self.lf - omega_gfm * ifd)
            ditd_dt = self.w_n * ((vfd - vgd - self.rt * itd) / self.lt + omega_gfm * itq)
            ditq_dt = self.w_n * ((vfq - vgq - self.rt * itq) / self.lt - omega_gfm * itd)
            dsigmad_dt = i_ref_d - ifd + self.kFFi * itd
            dsigmaq_dt = i_ref_q - ifq + self.kFFi * itq
            dgammaq_dt = self.wq * (self.s_qv * (self.q_ref - q_inst) - gamma_q)

            # Check se siamo in modalità batch (torch.Tensor) o singolo sample (numpy)
            is_torch = isinstance(xi_d, torch.Tensor)

            if is_torch:
                # ---- CASO TORCH (per training PINN con PyTorch) ----
                batch_size = xi_d.shape[0]

                if isinstance(self.w_n, torch.Tensor):
                    dthetagfm_dt = omega_gfm * self.w_n
                    dthetagrid_dt = self.w_n
                else:
                    dthetagfm_dt = torch.tensor(omega_gfm*self.w_n).to(self.device)
                    dthetagrid_dt = torch.tensor(self.w_n).to(self.device)

                # Espandi se necessario
                if dthetagfm_dt.dim() == 0:
                    dthetagfm_dt = dthetagfm_dt.expand(batch_size, 1)
                if dthetagrid_dt.dim() == 0:
                    dthetagrid_dt = dthetagrid_dt.expand(batch_size, 1)

                return [
                    dxi_d_dt, dxi_q_dt, dvfd_dt, dvfq_dt, difd_dt, difq_dt, ditd_dt, ditq_dt,
                    dsigmad_dt, dsigmaq_dt, dgammaq_dt, dthetagfm_dt, dthetagrid_dt
                ]

            else:
                # ---- CASO NUMPY (per integrazione con solve_ivp) ----
                dthetagfm_dt = omega_gfm * self.w_n
                dthetagrid_dt = self.w_n

                # Restituisci una lista di scalari o array NumPy 0D coerenti
                return np.array([
                    dxi_d_dt, dxi_q_dt, dvfd_dt, dvfq_dt, difd_dt, difq_dt, ditd_dt, ditq_dt,
                    dsigmad_dt, dsigmaq_dt, dgammaq_dt, dthetagfm_dt, dthetagrid_dt
                ], dtype=np.float64)


