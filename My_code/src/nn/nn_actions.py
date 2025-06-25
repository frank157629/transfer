import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.nn.nn_dataset import DataSampler
from src.nn.nn_model import Net, Network, PinnA, FullyConnectedResNet, Kalm
from src.functions import *
from src.nn.early_stopping import EarlyStopping
# from src.ode.sm_models_d import SynchronousMachineModels
from src.ode.ode import ODE

import wandb
import torch.autograd.functional as func
from src.nn.gradient_based_weighting import PINNWeighting
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


class NeuralNetworkActions():
    def __init__(self, cfg, modelling_full):  # The modelling equations are used, must be predefined, more choices to be added such as dynamic modelling
        self.cfg = cfg
        set_random_seeds(cfg.seed)  # set all seeds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.modelling_full = modelling_full
        self.data_loader = DataSampler(cfg)
        self.input_dim = self.data_loader.input_dim  # The input dimension is the number of input features
        self.output_dim = self.input_dim - 1  # The output dimension is the input dimension minus the time column

        self.model = self.define_nn_model()  # Create an instance of the class Net
        self.weight_init(self.model, cfg.nn.weight_init)  # Initialize the weights of the Net
        self.criterion = self.custom_loss(cfg.nn.loss_criterion)  # Define the loss function
        self.criterion_mae = nn.L1Loss()  # Define the MAE loss for testing
        self.optimizer = self.custom_optimizer(cfg.nn.optimizer, cfg.nn.lr)  # Define the optimizer
        self.scheduler = self.custom_learning_rate(cfg.nn.lr_scheduler)  # Define the learning rate scheduler

        # self.SynchronousMachineModels = SynchronousMachineModels(self.cfg) # Create an instance of the class SM_modelling
        self.ODE = ODE(self.cfg)  # Create an instance of the class SM_modelling
        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=cfg.nn.early_stopping_patience, verbose=True,delta=cfg.nn.early_stopping_min_delta)

    def pinn_train(self, wandb_run=None):
        num_of_skip_data_points = self.num_of_skip_data_points
        num_of_skip_col_points = self.num_of_skip_col_points
        num_of_skip_val_points = self.num_of_skip_val_points
        x_train, y_train, x_train_col, x_train_col_ic, y_train_col_ic, x_val, y_val = self.data_loader.define_train_val_data2(self.cfg.dataset.perc_of_data_points, self.cfg.dataset.perc_of_col_points, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points) # define the training and validation data

