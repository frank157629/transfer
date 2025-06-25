from torch.utils.data import Dataset
import pickle
import torch
import torch.nn as nn
import numpy as np
import os
import wandb
from omegaconf import OmegaConf
from pyDOE import lhs
from src.dataset.create_dataset_functions import ODE_modelling
from torch.utils.data import DataLoader, TensorDataset


class DataSampler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.idx = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_flag = cfg.model.model_flag  # 'SM_IB' or 'SM' or 'SM_AVR' or 'SM_GOV'
        self.shuffle = cfg.dataset.shuffle  # Define whether to shuffle the data before splitting
        self.split_ratio = cfg.dataset.split_ratio  # Define the ratio of the training set to the validation and test sets
        self.new_coll_points_flag = cfg.dataset.new_coll_points_flag  # Define whether to use new collocation points
        self.time = cfg.time  # Define the time limit for the data
        self.seed = cfg.model.seed
        self.data, self.input_dim, self.total_trajectories = self.load_data()  # Load the data from the file
        self.x, self.y = self.data_input_target_limited(self.data,self.time)  # Convert the loaded data into input and target data with a time limit
        self.sample_per_traj = self.x.shape[0] / self.total_trajectories  # number of samples per trajectory
        print("samples / traj =", self.sample_per_traj)
        # self.sample_per_traj = self.x.shape[0]/self.total_trajectories # number of samples per trajectory

        if self.cfg.dataset.validation_flag:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.train_val_test_split(self.x, self.y, self.split_ratio, True)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = self.train_val_test_split(self.x, self.y,self.split_ratio, False)
        if self.new_coll_points_flag:
            self.x_train_col = self.create_col_points().requires_grad_(True)
        else:
            self.x_train_col = self.x_train.clone().detach().requires_grad_(True)

        if self.cfg.dataset.transform_input != "None":
            self.minus_input, self.divide_input = self.define_minus_divide(self.x_train, self.x_train_col)
            self.minus_input = torch.nn.Parameter(self.minus_input, requires_grad=False)
            self.divide_input = torch.nn.Parameter(self.divide_input, requires_grad=False)
        if self.cfg.dataset.transform_output != "None":
            self.minus_target, self.divide_target = self.define_minus_divide(self.y_train, torch.empty(0))
            self.minus_target = torch.nn.Parameter(self.minus_target, requires_grad=False)
            self.divide_target = torch.nn.Parameter(self.divide_target, requires_grad=False)
