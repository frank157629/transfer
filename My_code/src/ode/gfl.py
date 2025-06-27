from omegaconf import OmegaConf
import os
import torch
import numpy as np



class GFL:
    def __init__(self, config):
        """
        Initialize the pll model with the given configuration.

        Parameters:
            config (dict): The configuration of the model.

        Attributes:
            params_dir (str): The path to the parameters directory.
            model_num (int): The number of the model to be used.
            model_flag (str): The model to be used.
            define the parameters of the model based on the model_num
        """
        self.params_dir = config.dirs.params_dir  # path to the parameters directory
        self.model_num = config.model.model_num  # the number of the model to be used
        self.model_flag = config.model.model_flag  # the model to be used
        self.define_model_params()  # define the parameters of the power system

    def define_model_params(self):
        """
        Define the parameters of the pll model based on the model_num
        and potentially the parameters of the AVR and the Governor.

        Returns:
            Attributes: The parameters of the pll model
        """
        # model_params_path = os.path.join(self.params_dir, "GFL_2nd_order" + str(self.model_num) + ".yaml")  # path to the selected model parameters
        # model_params = OmegaConf.load(model_params_path)
        yaml_file = f"{self.model_flag}{self.model_num}.yaml"
        model_params_path = os.path.join(self.params_dir, yaml_file)
        model_params = OmegaConf.load(model_params_path)
        if self.model_flag == 'TEST':
            for param in ['a']:
                setattr(self, param, getattr(model_params, param))

        if self.model_flag == 'GFL_2nd_order':
            for param in ['S_b','V_g','V_dc','f_0','w_g','T_s','X_Lg','R_Lg','L_Lg','X_R_ratio','i_d_c','i_q_c','K_p','K_i']:
                setattr(self, param, getattr(model_params, param))

        if self.model_flag == 'GFL_7th_order':
            for param in ["S_b", "V_g", "V_dc", "f_0", "w_g", "T_s", "L_g", "R_g", "C_f", "k_p", "k_i", "w_n", "v_gd_g", "v_gq_g", "i_Ld_g", "i_Lq_g"]:
                setattr(self, param, getattr(model_params, param))

        return



    def odequation_hl(self, t, x):
        """
        Calculates the derivatives of the state variables for the pll model.

        Parameters:
            t (float): The current time.
            x (list): The list of the two state variables

        Returns:
            list: A list of derivatives.
        """
        if self.model_flag == 'TEST':
            x = x
            if isinstance (x, torch.Tensor):
                dx_dt = self.a*x
            return [dx_dt]

        if self.model_flag == "GFL_2nd_order":
            delta, omega = x
            #define T_m, T_e, D
            if isinstance(delta, torch.Tensor):
                M = 1 - self.K_p * self.L_Lg * self.i_d_c
                T_m = self.K_i * (self.R_Lg * self.i_q_c + self.L_Lg * self.i_d_c * self.w_g)
                T_e = self.K_i * self.V_g * torch.sin(delta)
                D = self.K_p * self.V_g * torch.cos(delta) - self.K_i * self.L_Lg * self.i_d_c
            else:
                M = 1 - self.K_p * self.L_Lg * self.i_d_c
                T_m = self.K_i * (self.R_Lg * self.i_q_c + self.L_Lg * self.i_d_c * self.w_g)
                T_e = self.K_i * self.V_g * np.sin(delta)
                D = self.K_p * self.V_g * np.cos(delta) - self.K_i * self.L_Lg * self.i_d_c

            # Calculate delta derivative to time
            ddelta_dt = omega

            # Calculate omega derivative to time
            domega_dt = (T_m - T_e -D * omega ) * (1/M)
            return [ddelta_dt, domega_dt]

        if self.model_flag == "GFL_7th_order":
            gamma, delta, theta_pll, i_gd_g, i_gq_g, v_od_g, v_oq_g = x
            if isinstance(delta, torch.Tensor):
                dgamma_dt = -torch.sin(delta) * v_od_g + torch.cos(delta)*v_oq_g
                ddelta_dt = -self.k_p*torch.sin(delta)*v_od_g+self.k_p*torch.cos(delta)*v_oq_g+self.k_i*gamma+self.w_n-self.w_g
                dtheta_pll_dt = -self.k_p*torch.sin(delta)*v_od_g+self.k_p*torch.cos(delta)*v_oq_g+self.k_i*gamma+self.w_n
                di_gd_g_dt = (1/self.L_g)*(v_od_g-self.v_gd_g-self.R_g*i_gd_g+self.w_n*self.L_g*i_gq_g)
                di_gq_g_dt = (1/self.L_g)*(v_oq_g-self.v_gq_g-self.R_g*i_gq_g-self.w_n*self.L_g*i_gd_g)
                dv_od_g_dt = (1/self.C_f)*(self.i_Ld_g-i_gd_g+self.w_n*self.C_f*v_oq_g)
                dv_oq_g_dt = (1/self.C_f)*(self.i_Lq_g-i_gq_g-self.w_n*self.C_f*v_od_g)
            else:
                dgamma_dt = -np.sin(delta) * v_od_g + np.cos(delta) * v_oq_g
                ddelta_dt = -self.k_p * np.sin(delta) * v_od_g + self.k_p * np.cos(delta) * v_oq_g + self.k_i * gamma + self.w_n - self.w_g
                dtheta_pll_dt = -self.k_p * np.sin(delta) * v_od_g + self.k_p * np.cos(delta) * v_oq_g + self.k_i * gamma + self.w_n
                di_gd_g_dt = (1 / self.L_g) * (v_od_g - self.v_gd_g - self.R_g * i_gd_g + self.w_n * self.L_g * i_gq_g)
                di_gq_g_dt = (1 / self.L_g) * (v_oq_g - self.v_gq_g - self.R_g * i_gq_g - self.w_n * self.L_g * i_gd_g)
                dv_od_g_dt = (1 / self.C_f) * (self.i_Ld_g - i_gd_g + self.w_n * self.C_f * v_oq_g)
                dv_oq_g_dt = (1 / self.C_f) * (self.i_Lq_g - i_gq_g - self.w_n * self.C_f * v_od_g)

            return [dgamma_dt, ddelta_dt, dtheta_pll_dt, di_gd_g_dt, di_gq_g_dt, dv_od_g_dt, dv_oq_g_dt]












