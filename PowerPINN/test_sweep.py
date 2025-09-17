"PowerPINN/test_sweep.py"
"""
Entry point for launching W&B hyperparameter sweeps.

- Loads sweep configuration (PINN or Vanilla NN) from `config.yaml`
- Dynamically adjusts training settings per sweep run
- Supports both Physics-Informed Neural Networks and Vanilla NNs
- Initializes and runs the W&B sweep agent
"""
import os

from ode.gfl_models_d import GridFollowingConverterModels
from src.nn.pinn_actions import PhysicsInformedNeuralNetworkActions
from src.nn.vanilla_actions import VanillaNeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb

def train(config=None):
    run = wandb.init(config=config)
    config = run.config

    setup = OmegaConf.load("src/conf/config.yaml")

    # Load base configuration from YAML
    if setup.theme == "GFL":
        if setup.train == "pinn":
            cfg = OmegaConf.load("src/conf/setup_dataset_pinn_gfl.yaml")
        elif setup.train == "vanilla":
            cfg = OmegaConf.load("src/conf/setup_dataset_vanilla_gfl.yaml")
    else:
        raise NotImplementedError

    cfg.seed = config.seed
    cfg.network.weighting.weights = [config.weight_data, config.weight_dt, config.weight_pinn, config.weight_pinn_ic]
    cfg.theme = setup.theme
    cfg.dataset.number = config.number
    cfg.network.optimizer = config.optimizer
    cfg.dataset.new_coll_points_flag = config.new_coll_points_flag
    cfg.dataset.batch_size = config.batch_size

    if cfg.network.optimizer == "LBFGS":
        lbfgs_iter = 10
        cfg.network.early_stopping_patience = int(cfg.network.early_stopping_patience / lbfgs_iter)
        cfg.network.num_epochs = int(cfg.network.num_epochs / lbfgs_iter)
        cfg.network.weighting.update_weights_freq = int(cfg.network.weighting.update_weights_freq*4) # increase due to internal iterations, around 25 internal iterations per epoch


    # Initialize model and network
    if cfg.theme == "GFL":
        modelling_full = GridFollowingConverterModels(cfg)
        if setup.train == "pinn":
            pinn = PhysicsInformedNeuralNetworkActions(cfg, modelling_full)
        elif setup.train == "vanilla":
            vanilla = VanillaNeuralNetworkActions(cfg)

    #Haitian, Skip point configs are now handled internally in `pinn_train2()` instead of being passed as arguments
    if setup.train == "pinn":
        print("PINN")
        pinn.pinn_train2(run)
    elif setup.train == "vanilla":
        print("VANILLA")
        vanilla.vanilla_train(run)



    run.finish()

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "Test_loss",
            "goal": "minimize"
        }
        #Configuration for Vanilla NN sweep
        , "parameters": {
            "number": {"values": [1]},
            "seed": {"values": [1]},
            "weight_data": {"values": [1]},
            "weight_dt": {"values": [0]},
            "weight_pinn": {"values": [0]},
            "weight_pinn_ic": {"values": [0]},
            "optimizer": {"values": ["Adam"]},
            "new_coll_points_flag": {"values": [False]},
            "batch_size": {"values": ["None"]},
        }

        ## Alternative configuration for PINN sweep
        # , "parameters": {
        #     "number": {"values": [1]},
        #     "seed": {"values": [1]},
        #     "weight_data": {"values": [1]},
        #     "weight_dt": {"values": [1e-3]},
        #     "weight_pinn": {"values": [1e-4]},
        #     "weight_pinn_ic": {"values": [1e-1]},
        #     "optimizer": {"values": ["Adam"]},
        #     "new_coll_points_flag": {"values": [False]},
        #     "batch_size": {"values": ["None"]},
        # }
    }

    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep_config, project="PINN-ΚΑΝ")
    wandb.agent(sweep_id, function=train)
