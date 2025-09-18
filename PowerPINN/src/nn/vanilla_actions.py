"PowerPINN/src/nn/vanilla_actions.py"
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.nn.nn_dataset import DataSampler
from src.nn.nn_model import Network
from src.functions import *
from src.nn.early_stopping import EarlyStopping
from src.ode.gfl_models_d import GridFollowingConverterModels
import wandb
import torch.autograd.functional as func
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from src.ode.gfl_models_d import calculate_frequency

class VanillaNeuralNetworkActions():
    """
    A class used to define the actions of the Vanilla NN

    Attributes
    ----------
    cfg (dict) : configuration file
    input_dim (int) : number of input features
    hidden_dim (int) : number of hidden neural network layers
    output_dim (int) : number of output features
    learning_rate (float) : learning rate of the optimizer
    model (Net) : neural network model class
    criterion (nn.Module) : loss function
    optimizer (optim) : optimizer
    scheduler (optim) : learning rate scheduler
    GFL_model (GFL_modelling) : class for creating the GFL model
    machine_params (dict) : parameters of the GFL
    system_params (dict) : parameters of the power system
    modelling_eq (CreateSolver) : class for solving the GFL model
    flag_for_modelling (bool) : flag for using the GFL model
    device (torch.device) : device to run the model

    Methods
    -------
    define_nn_model()
        This function defines the neural network model
    custom_loss(loss_name)
        This function defines the loss function
    custom_optimizer(optimizer_name, learning_rate)
        This function defines the optimizer
    custom_learning_rate(lr_name)
        This function defines the learning rate scheduler
    weight_init(module, init_name)
        This function initializes the weights of the neural network model
    test(x_test)
        This function tests the neural network model
    plot(x_train, y_train, var=0)
        This function plots the data for a specific variable
    plot_all(x_train, y_train)
        This function plots all the data in pairs
    forward_nn(time, no_time)
        This function calculates the output of the neural network model, input is given as time and the other input columns
    forward_pass(x_train)
        This function calculates the output of the neural network model, input is given as the one whole tensor
    """
    def __init__(self, cfg): # The modelling equations are used, must be predefined, more choices to be added such as dynamic modelling
        self.cfg = cfg
        set_random_seeds(cfg.seed) # set all seeds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_loader = DataSampler(cfg)
        self.input_dim = self.data_loader.input_dim # The input dimension is the number of input features
        self.output_dim = self.input_dim-1 # The output dimension is the input dimension minus the time column
        self.model_flag = cfg.model.model_flag  # the model to be used
        self.model = self.define_vanilla_model()  # Create an instance of the class Net
        self.weight_init(self.model, cfg.network.weight_init) # Initialize the weights of the Net
        self.criterion = self.custom_loss(cfg.network.loss_criterion) # Define the loss function
        self.criterion_mae = nn.L1Loss() # Define the MAE loss for testing
        self.optimizer = self.custom_optimizer(cfg.network.optimizer, cfg.network.lr) # Define the optimizer
        self.scheduler = self.custom_learning_rate(cfg.network.lr_scheduler) # Define the learning rate scheduler

        # Create an instance of the class xxx_modelling
        if self.cfg.theme == "GFL":
            self.GridFollowingConverterModels = GridFollowingConverterModels(self.cfg)

        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=cfg.network.early_stopping_patience, verbose=True, delta=cfg.network.early_stopping_min_delta)

    def setup_nn(self):
        self.model = self.define_vanilla_model()  # Create an instance of the class Net
        self.weight_init(self.model, self.cfg.network.weight_init) # Initialize the weights of the Net
        self.criterion = self.custom_loss(self.cfg.network.loss_criterion) # Define the loss function
        if self.cfg.network.optimizer == "Hybrid":  # Define the optimizer
            self.optimizer = self.custom_optimizer("Adam", self.cfg.network.lr)
            self.optimizer2 = self.custom_optimizer("LBFGS", self.cfg.network.lr)
        else:
            self.optimizer = self.custom_optimizer(self.cfg.network.optimizer, self.cfg.network.lr)
        self.scheduler = self.custom_learning_rate(self.cfg.network.lr_scheduler) # Define the learning rate scheduler
        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=self.cfg.network.early_stopping_patience, verbose=True, delta=self.cfg.network.early_stopping_min_delta)
        if self.cfg.network.update_weight_method=="ReLoBRaLo":
            self.relobralo_loss = ReLoBRaLoLoss()
        return

    #Haitian, define vanilla NN
    def define_vanilla_model(self):
        """
        This function defines the neural network model
        """
        print("Selected deep learning model: ",self.cfg.network.type)

        if self.cfg.network.type == "DynamicNN": # Dynamic architecture of the neural network
            model = Network(self.input_dim, self.cfg.network.hidden_dim, self.output_dim, self.cfg.network.hidden_layers)
        else:
            raise Exception("NN type not found")
        return model

    def custom_loss(self, loss_name):
        """
        This function defines the loss function

        Args:
            loss_name (str): name of the loss function

        Returns:
            criterion (nn.Module): loss function
        """
        if loss_name == 'MSELoss': # Mean Squared Error Loss
            criterion = nn.MSELoss()
        elif loss_name == 'L1Loss': # Mean Absolute Error Loss
            criterion = nn.L1Loss()
        elif loss_name == 'SmoothL1Loss': # Huber Loss
            criterion = nn.SmoothL1Loss()
        else:
            raise Exception("Loss not found")
        return criterion

    def custom_optimizer(self, optimizer_name, learning_rate):
        """
        This function defines the optimizer

        Args:
            optimizer_name (str): name of the optimizer
            learning_rate (float): learning rate of the optimizer

        Returns:
            optimizer (optim): optimizer
        """
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'Adam_decay':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0001)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'LBFGS':
            optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        else:
            raise Exception("Optimizer not found")
        return optimizer

    def custom_learning_rate(self, lr_name): # Choose between "StepLR", "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau
        """
        This function defines the learning rate scheduler

        Args:
            lr_name (str): name of the learning rate scheduler

        Returns:
            scheduler (optim): learning rate scheduler
        """
        if lr_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        elif lr_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,10000], gamma=0.1)
        elif lr_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif lr_name == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif lr_name == 'No_scheduler':
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1)
        else:
            raise Exception("Learning rate not found")
        return scheduler

    def weight_init(self,module, init_name):
        """
        This function initializes the weights of the neural network model

        Args:
            module (Net): neural network model
            init_name (str): name of the initialization method
        """
        for m in module.modules():
            if type(m) == nn.Linear:
                if init_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init_name == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                elif init_name == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_name == 'normal':
                    pass
                else:
                    raise Exception("Initialization not found")
        return

    def test(self, x_test):
        """
        This function tests the neural network model

        Args:
            x_test (torch.Tensor): input data

        Returns:
            y_pred (torch.Tensor): predicted output data
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = self.forward_pass(x_test)
        return y_pred

    def forward_nn(self, time, no_time):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        x_train = torch.cat((time, no_time), 1)
        y_pred = self.model.forward(x_train)
        y_pred = self.data_loader.detransform_output(y_pred)
        if self.cfg.network.type == "DynamicNN":
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')

    def forward_pass(self, x_train):
        """
        This function calculates the output of the neural network model, input is given as time and the other input columns
        """
        time = x_train[:,0].unsqueeze(1) # get the time column
        no_time = x_train[:,1:]
        y_pred = self.model.forward(x_train)
        y_pred = self.data_loader.detransform_output(y_pred)
        if self.cfg.network.type == "DynamicNN":
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')

    def folder_name_f2(self,cfg):
        weight_data, weight_dt, weight_pinn, weight_pinn_ic = cfg.network.weighting.weights

        self.weight_data = 1
        self.weight_dt = 0
        self.weight_pinn = 0
        self.weight_pinn_ic = 0

        if weight_data == weight_dt == 0:
            if weight_pinn >0 and weight_pinn_ic > 0:
                folder_name = "pinn_ic"
            elif weight_pinn > 0:
                folder_name = "pinn"
        if weight_dt == weight_pinn == 0:
            if weight_data > 0:
                folder_name = "data_ic" if weight_pinn_ic > 0 else "data" #only data
        elif weight_data == 0:
            if weight_dt > 0 and weight_pinn > 0:
                folder_name = "dt_pinn" if weight_pinn_ic == 0 else None #only ode loss
        elif weight_dt > 0 and weight_pinn > 0:
            folder_name = "data_dt_pinn_ic" if weight_pinn_ic > 0 else "data_dt_pinn" #all or no pinn_ic
        elif weight_dt > 0 and weight_pinn == weight_pinn_ic == 0:
            folder_name = "data_dt" #only data and dt
        elif weight_data == 0 and weight_pinn > 0:
            folder_name = "pinn_ic" if weight_pinn_ic > 0 else "pinn" # pinn_ic or only collocation- pinn loss
        else:
            raise Exception("Folder name not found")

        if not folder_name:
            raise Exception("Folder name not found")

        return folder_name

    def initialize_loss_weights(self, weight_data, weight_dt, weight_pinn, weight_pinn_ic):

        self.weight_data = 1
        self.weight_dt = 0
        self.weight_pinn = 0
        self.weight_pinn_ic = 0
        return

    def calc_adapt_criterion_loss(self, x_train, y_train, output):
        """
        This function calculates the loss with the adaptive criterion
        """
        if self.cfg.network.time_factored_loss == True:
            time = x_train[:,0].unsqueeze(1) # get the time column
            end_time = self.cfg.time
            end_time = torch.tensor([end_time]).to(self.device)
            time_factor = end_time - time
            self.criterion2 = self.criterion.__class__(reduction='none')
            # Calculate element-wise L1 loss and then scale by time factor
            base_loss = self.criterion2(y_train, output) * time_factor # L1 loss
            #base_loss = base_loss  # Scale by time factor
            # calculate the mean of the loss per column
            loss_list = [base_loss[:, i].mean() for i in range(y_train.shape[1])]
            return base_loss.mean(), loss_list # Aggregate the loss
        else:
            loss_list = [self.criterion(y_train[:, i], output[:, i]) for i in range(y_train.shape[1])]
            return torch.mean(torch.stack(loss_list)), loss_list

    #Haitian, by passing the skip points through function parameters, now changed to passing within the function using .yaml
    def vanilla_train(self,wandb_run=None):
        """
        This function trains the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        num_of_skip_data_points = self.cfg.network.num_of_skip_data_points
        num_of_skip_val_points  = self.cfg.network.num_of_skip_val_points

        ret = self.data_loader.define_train_val_data2(
            self.cfg.dataset.perc_of_data_points,  # perc_of_data_points
            0,                                     # perc_of_col_points = 0
            num_of_skip_data_points,               # num_of_skip_data_points
            1,                                     # num_of_col_points
            num_of_skip_val_points                 # num_of_skip_val_points
        )

        x_train, y_train, x_val, y_val = ret[0], ret[1], ret[-2], ret[-1]

        batch_size = self.cfg.network.batch_size if self.cfg.network.batch_size != "None" else len(x_train)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

        folder_name=self.folder_name_f2(self.cfg)
        os.makedirs(os.path.join(self.cfg.dirs.model_dir, folder_name),exist_ok=True)
        self.wandb_run = wandb_run

        print("getting in training")
        for epoch in range(self.cfg.network.num_epochs):


            self.model.train() # set the model to training mode
            if self.cfg.network.batch_size != "None":

                for x_batch, y_batch in train_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                    self.optimizer.zero_grad()
                    output = self.forward_pass(x_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    # 记录（只用 data）
                    self.loss_total = loss
                    self.loss_data  = loss
            else:
                def closure():

                    self.optimizer.zero_grad()
                    output = self.forward_pass(x_train)
                    loss = self.criterion(output, y_train)
                    loss.backward()
                    self.optimizer.step()

                    self.loss_total = loss
                    self.loss_data  = loss

                self.optimizer.step(closure)

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val_dev = x_val.to(self.device)
                y_val_dev = y_val.to(self.device)
                val_outputs = self.forward_pass(x_val_dev)
                loss_val_data = self.criterion(val_outputs, y_val_dev)

            val_loss = loss_val_data.item()

            if (epoch + 1) % self.cfg.network.weighting.update_weights_freq == 0:
                if self.cfg.network.weighting.update_weight_method=="Sam":
                    self.weighting_scheme.update_weights(self.losses, epoch)

                # log some plots to wandb
                if wandb_run is not None:
                    self.log_plot(val_outputs, y_val, epoch, wandb_run, x_val,"validation", 0, 500)

            if (epoch + 1 ) % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg.network.num_epochs}], Loss: {self.loss_total.item():.4f}', val_loss)

            # log all the losses for the epoch to wandb
            save_iteration = 500 if self.cfg.network.optimizer == "LBFGS" else 10000 # 20 iterations within the optimizer ->500*20 = 10000
            if (epoch + 1) % save_iteration == 0:

                name = f"{self.cfg.model.model_flag}{self.cfg.network.type}_{self.cfg.time}_{epoch+1}_{self.data_loader.training_shape}_{self.data_loader.training_col_shape}_{self.data_loader.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_{self.cfg.network.weighting.update_weight_method}.pth"

                self.save_model(os.path.join(folder_name, name))

                if wandb_run is not None:
                    log_data = {
                        "Val_loss": val_loss,
                        "Loss": self.loss_total.item(),
                        "Loss_data": self.loss_data.item(),
                        "epoch": epoch
                    }
                    wandb_run.log(log_data)


            if self.cfg.network.early_stopping:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    self.early_stopping.save_checkpoint(val_loss, self.model)
                    break

        if self.early_stopping.early_stop == True or (epoch + 1) % save_iteration != 0:
            name = f"{self.cfg.model.model_flag}{self.cfg.network.type}_{self.cfg.time}_{epoch+1}_{self.data_loader.training_shape}_{self.data_loader.training_col_shape}_{self.data_loader.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_{self.cfg.network.weighting.update_weight_method}.pth"
            self.save_model(os.path.join(folder_name, name))
        self.final_name = os.path.join(folder_name, f"{self.cfg.model.model_flag}{self.cfg.network.type}_{self.cfg.time}_{epoch+1}_{self.data_loader.training_shape}_{self.data_loader.training_col_shape}_{self.data_loader.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_{self.cfg.network.weighting.update_weight_method}")
        total_test_loss =  self.test_model(0,500,wandb_run)
        return

    #Haitian, change log_plot
    def test_model(self, starting_traj=0, total_traj=1, run=None):
        """
        Tests the trained model using test trajectories and logs test metrics.

        Args:
            starting_traj (int): Index of the first test trajectory to evaluate.
            total_traj (int): Number of test trajectories to evaluate.
            run (wandb.Run, optional): W&B run object for logging.
        """

        total_traj = total_traj if total_traj < self.data_loader.total_test_trajectories else self.data_loader.total_test_trajectories
        sample_per_traj = int(self.data_loader.sample_per_traj)

        x_test,y_test = self.data_loader.define_test_data(starting_traj,sample_per_traj,total_traj)
        self.model.eval()
        y_pred = self.forward_pass(x_test)
        test_loss = self.criterion(y_pred, y_test)
        print("Total test trajectories",total_traj)
        print(f'Loss: {test_loss.item():.8f}')
        test_loss_mae = self.criterion_mae(y_pred, y_test)
        print(f'MAE Loss: {test_loss_mae.item():.8f}')
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total trainable parameters',total_trainable_params)
        print("run", run)
        if run is not None:
            run.log({"Test_loss": test_loss.item() })
            run.log({"MAE Test loss": test_loss_mae.item() })
            # self.log_plot(y_pred, y_test, None, run,x_test)
            self.log_plot(y_pred, y_test, None, run, x_test,"test", starting_traj, total_traj)
        mae, rmse = self.loss_over_time(x_test, y_test, y_pred, run)
        return test_loss.item()

    # Haitian, added different logic for plotting GFL
    def log_plot(self, output, target, epoch, run, x_test, type, starting_traj=0, total_traj=1):
        # log in wandb

        if self.cfg.theme == "GFL":
            modeling_guide_path = os.path.join(self.cfg.dirs.init_conditions_dir, "modellings_guide_gfl.yaml")
            modeling_guide = OmegaConf.load(modeling_guide_path)

        # check if proposed modeling is in the modeling guide
        for model in modeling_guide:
            model_name = model.get("name")
            if model_name == self.cfg.model.model_flag:
                self.keys = model.get("keys")

        pts_per_traj = int(self.data_loader.sample_per_traj)
        max_traj = len(output) // pts_per_traj
        print("type: ",str(type) + ", total_traj:" , str(max_traj) + ", max_traj", str(total_traj))
        total_traj = min(total_traj, max_traj)

        blk_idx = list(range(starting_traj, starting_traj + total_traj))
        fig, axes = plt.subplots(total_traj, 3,figsize=(27, 3 * total_traj),sharex='col')

        for r, k in enumerate(blk_idx):
            lo, hi = k * pts_per_traj, (k + 1) * pts_per_traj       #Slice current trajectory
            t = x_test[lo:hi, 0].detach().cpu().numpy()
            delta_true = target[lo:hi, 0].detach().cpu().numpy()
            omega_true = target[lo:hi, 1].detach().cpu().numpy()
            delta_pred = output[lo:hi, 0].detach().cpu().numpy()
            omega_pred = output[lo:hi, 1].detach().cpu().numpy()
            #Calculate frequency
            f_true = calculate_frequency(omega_true, np.pi * 100)
            f_pred = calculate_frequency(omega_pred, np.pi * 100)

            # --- δ ---
            if self.keys[0] == "delta":
                axd = axes[r, 0]
                axd.set_visible(True)
                axd.plot(t, delta_true, color='tab:blue', lw=1.2,label='True' if r == 0 else None)
                axd.plot(t, delta_pred, color='tab:orange', lw=1.2, ls='--',label='Pred' if r == 0 else None)
                axd.grid(ls='--', alpha=.3)

            if self.keys[1] == "omega":
                # --- ω ---
                axw = axes[r, 1]
                axw.set_visible(True)
                axw.plot(t, omega_true, color='tab:blue', lw=1.2,
                         label='True' if r == 0 else None)
                axw.plot(t, omega_pred, color='tab:orange', lw=1.2, ls='--',
                         label='Pred' if r == 0 else None)
                axw.grid(ls='--', alpha=.3)

                # --- f ---
                axf = axes[r, 2]
                axf.set_visible(True)
                axf.plot(t, f_true, color='tab:blue', lw=1.2,label='True' if r == 0 else None)
                axf.plot(t, f_pred, color='tab:orange', lw=1.2, ls='--',label='Pred' if r == 0 else None)
                axf.grid(ls='--', alpha=.3)

            # Row labelling
            axd.text(-0.05, 0.5, f'traj {k+1}',transform=axd.transAxes,va='center', ha='right',fontsize=9, weight='bold')

        # label x-axes
        axes[0, 0].set_title('δ')
        axes[0, 1].set_title('ω')
        axes[0, 2].set_title('f = ω/2*pi')
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        axes[-1, 2].set_xlabel('Time (s)')

        #??
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')

        plt.tight_layout()

        # --------------------------- wandb docu.
        gname = f"{type}{blk_idx[0]+1}-{blk_idx[-1]+1}"  # Example: 1-5, 6-10 …
        run.log({f"traj_{gname}": wandb.Image(fig)},commit=False)

        plt.close(fig)
        run.log({}, commit=True)



    def loss_over_time(self, x_test, y_test, y_pred, run = None):

        unique_values = torch.unique(x_test[:,0]) # get the unique values of the time
        mae = []
        rmse = []
        for value in unique_values: # for each time step
            index = torch.where(x_test[:,0] == value) # find the indexes of the time step
            # calculate the mae and rmse for each value
            y_pred_ = y_pred[index] # keep only the points at the specific time
            y_true = y_test[index] # keep only the points at the specific time
            mae_var = []
            rmse_var = []
            for i in range(y_pred_.shape[1]):
                mae_var.append(self.criterion_mae(y_pred_[:,i], y_true[:,i]).item()) # calculate the mae for each variable
                rmse_var.append(self.criterion(y_pred_[:,i], y_true[:,i]).item()) # calculate the rmse for each variable
            mae.append((mae_var))
            rmse.append((rmse_var))
        mae = np.array(mae)
        rmse = np.array(rmse)
        mae2 = torch.abs(y_test - y_pred)  # Calculate absolute errors for each prediction

        if self.cfg.theme == "GFL":
            if self.model_flag == "GFL_2nd_order":
                var_name = ["delta", "omega"]
        else:
            raise NotImplementedError
        if run is not None:
            for i in range(y_pred_.shape[1]):
                mean_mae = np.mean(mae[:,i])
                mean_rmse = np.mean(rmse[:,i])
                max_mae = torch.max(mae2[:,i])  # Find the maximum absolute error # calculate the absolute error for each prediction, to find max mae

                #log only mean values
                run.log({f"Mean MAE for variable {self.keys[i]}": mean_mae})
                run.log({f"Mean MSE for variable {self.keys[i]}": mean_rmse})
                run.log({f"Max MAE for variable {self.keys[i]}": max_mae.item()})
            time = unique_values.detach().cpu().numpy()
            for i in range(y_pred_.shape[1]):
                for j in range(time.shape[0]):
                    run.log({f"MAE for variable {self.keys[i]}": mae[j,i], "Time": time[j]})
                    run.log({f"MSE for variable {self.keys[i]}": rmse[j,i], "Time": time[j]})

        max_mae = torch.max(mae2)  # Find the maximum absolute error
        run.log({"Test Max AE": max_mae.item()})
        #save the mae and rmse
        full_path = os.path.join(self.cfg.dirs.model_dir, self.final_name)
        np.save(full_path+"_mae.npy", mae)
        np.save(full_path+"_mse.npy", rmse)

        return mae, rmse

    def save_model(self,name):
        """
        Save model weights to the model_dir.

        Args:
            name (str): name of the model
        """
        #save model to the model_dir
        model_dir = self.cfg.dirs.model_dir
        #find if there is such folder in the model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir,name)


        model_data = {"model_state_dict":self.model.state_dict()}
        if self.cfg.dataset.transform_input != "None":
            #extend the model_data dict
            model_data["minus_input"] = self.data_loader.minus_input
            model_data["divide_input"] = self.data_loader.divide_input

        if self.cfg.dataset.transform_output != "None":
            model_data["minus_target"] = self.data_loader.minus_target
            model_data["divide_target"] = self.data_loader.divide_target

        torch.save(model_data, model_path)

        print("Model( and tf values) saved:", model_path)
        return

    def load_model(self,name=None):
        """
        Load neural network model weights from the model_dir.

        Args:
            name (str): name of the model
        """
        #load model from the model_dir
        model_dir = self.cfg.dirs.model_dir
        if not os.path.exists(model_dir) or len(os.listdir(model_dir))==0:
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
        if name is None:
            #find first model in the model_dir
            name=os.listdir(model_dir)[0]
            if name=='.gitkeep':
                if len(os.listdir(model_dir))==1:
                    raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")
                name=os.listdir(model_dir)[1]
            print("load model:",name)

        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            print(os.path.join(model_dir,name))
            raise Exception("No model found in the model_dir, please correct path and name of the model from the bucket first")

        model_data = torch.load(model_path)
        self.model.load_state_dict(model_data['model_state_dict'])
        return None



    def plot(self, x_train, y_train, var=0):
        """
        This function plots the data

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            var (int): variable to plot
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy() # x is the time
        y_train = y_train[:,var].cpu().detach().numpy()
        y_pred = y_pred[:,var].cpu().detach().numpy()
        plt.figure()
        plt.plot(x_train, y_train, 'ro', label='Original data')
        plt.plot(x_train, y_pred, 'kx-', label='Fitted line')
        plt.show()
        return

    def plot_all(self, x_train, y_train):
        """
        This function plots all the data in pairs

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
        """
        y_pred = self.test(x_train)
        x_train = x_train[:,0].cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        plt.figure(figsize=(10, 5))  # Create a figure with a specific size
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)  # Create subplots, alternating between two columns
            plt.plot(x_train, y_train[:, i], 'ro', label='Original data')
            plt.plot(x_train, y_pred[:, i], 'kx-', label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()  # Show the plot after every two iterations
        return

    def plot_all_dt(self, x_train, y_train):
        """
        This function plots the derivative of all the data in pairs
        """
        y_pred = self.test(x_train) # Predict the output data
        x_train = x_train[:,0].cpu().detach().numpy() # Keep only the time column
        dt = self.calculate_from_ode(y_train)  # Calculate the derivative of the output data
        dt_pred = self.calculate_from_ode(y_pred) # Calculate the derivative of the predicted output data
        dt = dt.cpu().detach().numpy()
        dt_pred = dt_pred.cpu().detach().numpy()
        plt.figure(figsize=(10, 5))
        for i in range(y_train.shape[1]):
            plt.subplot(1, 2, i % 2 + 1)
            plt.plot(x_train, dt[:, i], 'ro', label='Original data')
            plt.plot(x_train, dt_pred[:, i], 'kx-', label='Fitted line')
            plt.legend()
            if i % 2 != 0:
                plt.show()
        return