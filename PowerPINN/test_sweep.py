import os

from ode.gfl_models_d import GridFollowingConverterModels
from src.ode.sm_models_d import SynchronousMachineModels
from src.nn.pinn_actions import PhysicsInformedNeuralNetworkActions
from src.functions import *
from omegaconf import OmegaConf
import wandb

def train(config=None):
    run = wandb.init(config=config)
    config = run.config

    setup = OmegaConf.load("src/conf/config.yaml")

    # Load base configuration from YAML

    if setup.theme == "SM":
        cfg = OmegaConf.load("src/conf/setup_dataset_nn_sm.yaml")
    elif setup.theme == "GFL":
        cfg = OmegaConf.load("src/conf/setup_dataset_nn_gfl.yaml")
    elif setup.theme == "GFM":
        #Add your specifications here...
        raise NotImplementedError
    else:
        raise NotImplementedError

    cfg.seed = config.seed
    cfg.nn.weighting.weights = [config.weight_data, config.weight_dt, config.weight_pinn, config.weight_pinn_ic]
    cfg.theme = setup.theme

    if cfg.nn.optimizer == "LBFGS":
        lbfgs_iter = 10
        cfg.nn.early_stopping_patience = int(cfg.nn.early_stopping_patience / lbfgs_iter)
        cfg.nn.num_epochs = int(cfg.nn.num_epochs / lbfgs_iter)
        cfg.nn.weighting.update_weights_freq = int(cfg.nn.weighting.update_weights_freq*4) # increase due to internal iterations, around 25 internal iterations per epoch


    # Initialize model and network
    if cfg.theme == "SM":
        modelling_full = SynchronousMachineModels(cfg)
        network = NeuralNetworkActions(cfg, modelling_full)
    elif cfg.theme == "GFL":
        modelling_full = GridFollowingConverterModels(cfg)
        network = PhysicsInformedNeuralNetworkActions(cfg, modelling_full)
    elif cfg.theme == "GFM":
        ##Add your specifications here...
        raise NotImplementedError


    # Set skip points and start training
    #Haitian, changed pinn_train2, removed passing the skip_points as parameters of function but inside the function.
    network.pinn_train2(run)
    run.finish()

if __name__ == "__main__":
    # Define sweep configuration
    sweep_config = {
        "method": "grid",
        "metric": {
            "name": "Test_loss",
            "goal": "minimize"
        },
        "parameters": {
            "seed": {"values": [1]},
            "weight_data": {"values": [1]},
            "weight_dt": {"values": [1e-3]},
            "weight_pinn": {"values": [1e-4]},
            "weight_pinn_ic": {"values": [1e-3]}
        }
    }

    # Initialize and run sweep
    sweep_id = wandb.sweep(sweep_config, project="PINN-ΚΑΝ")
    wandb.agent(sweep_id, function=train)
