import numpy as np
import matplotlib.pyplot as plt
import omegaconf as OmegaConf
import os
from src.dataset.create_dataset_functions import create_init_conditions_set3
class test_functions:
    def __init__(self, config):
        self.seed = config.seed
        self.init_conds

    def shape(self, sol):
        self.shape = sol.shape


    def check_init_conditions(self, sampled_init_conds, indices):
        created_init_conds = create_init_conditions_set3()
        for i in indices:
            print
            if sampled_init_conds[i] != created_init_conds[i]:
                print("Index: ", indices, ", initial value doesn't match with the .pkl file.")
            else:
                print("Index: ", indices, ", initial value matches the .pkl file.")

        return
