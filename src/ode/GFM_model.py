import numpy as np
from scipy.integrate import solve_ivp
from src.functions import set_time
import torch
import os
from omegaconf import OmegaConf


class GFM:
    def __init__(self, config):
        self.params_dir = config.dirs.params_dir
        self.machine_num = config.model.machine_num
        self.model_flag = config.model.model_flag
        self.define_machine_params()  # define parameters of GFM
        self.define_system_params()  # define parameters of grid

        # Definisci V_ref e P_ref se ti servono costanti
        self.v_ref = 1.0
        self.p_ref = 1200e3 / self.S_n
        self.q_ref = 700e3 / self.S_n

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
            dthetagfm_dt = omega_gfm * self.w_n
            dthetagrid_dt = self.w_n

            return [
                dxi_d_dt, dxi_q_dt, dvfd_dt, dvfq_dt, difd_dt, difq_dt, ditd_dt, ditq_dt,
                dsigmad_dt, dsigmaq_dt, dgammaq_dt, dthetagfm_dt, dthetagrid_dt
            ]
