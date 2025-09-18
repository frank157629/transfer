"PowerPINN/src/ode/gfl_models_d.py"
from omegaconf import OmegaConf
import os
import torch
import numpy as np

#Haitian, Global time variable
_last_check_time = None
import time

def calculate_frequency(delta_omega, w_g):
    omega = delta_omega + w_g
    f = omega / (2 * np.pi)
    return f

#Haitan, to discover errors
def check_state_sanity(t, x, den=None, max_step_time=10):
    global _last_check_time
    now = time.time()

    # check runtime
    if _last_check_time is not None:
        duration = now - _last_check_time
        if duration > max_step_time:
            raise RuntimeError(f" Step at t={t:.4f} took too long: {duration:.3f} s")
    _last_check_time = now

    # 2. check denominator
    if den is not None:
        # if a single value, cat to list
        den_list = den if isinstance(den, (list, tuple, np.ndarray)) else [den]
        for d in den_list:
            if abs(d) < 1e-6:
                raise ZeroDivisionError(f"den≈0 at t={t:.4f}, state= {x}, den={d}")
    # 3. check exceptions
    if np.isnan(x).any():
        raise RuntimeError(f"NaN detected at t={t:.4f}, state={x}")
    if np.isinf(x).any():
        raise RuntimeError(f"Inf detected at t={t:.4f}, state={x}")
    if np.max(np.abs(x)) > 1e6:
        raise RuntimeError(f"Explosion: |x| > 1e6 at t={t:.4f}, state={x}")

#Haitian, the file are newly defined by Haitian to suit for the GFL needs.
class GridFollowingConverterModels:
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
        self.define_system_params()

    def define_system_params(self):
        return



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
        if self.model_flag == 'GFL_2nd_order':
            self.model_name = "Reduced-order model for GFL_2nd_order control with two states"
            for param in ['S_b','V_g','V_dc','f_0','w_g','T_s','X_Lg','R_Lg','L_Lg','X_R_ratio','i_d_c','i_q_c','K_p','K_i']:
                setattr(self, param, getattr(model_params, param))

        return
    #Haitian, to prevent faults during solving the ODE
    import numpy as np
    import time



    def odequation_gfl(self, t, x):
        """
        Calculates the derivatives of the state variables for the pll model.

        Parameters:
            t (float): The current time.
            x (list): The list of the two state variables

        Returns:
            list: A list of derivatives.
        """
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

        else:
            raise NotImplementedError


