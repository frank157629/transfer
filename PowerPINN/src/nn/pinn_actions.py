import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.nn.nn_dataset import DataSampler
from src.nn.nn_model import Net, Network,PinnA, FullyConnectedResNet, Kalm
from src.functions import *
from src.nn.early_stopping import EarlyStopping
from src.ode.sm_models_d import SynchronousMachineModels
from src.ode.gfl_models_d import GridFollowingConverterModels
import wandb
import torch.autograd.functional as func
from src.nn.gradient_based_weighting import PINNWeighting
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from src.ode.gfl_models_d import calculate_frequency



'''
    Haitian, this was the nn_actions from the toolbox originally. 
    However, we are implementing training logic here only for pinn, the vanilla NN is defined in the same folder as vanilla_actions
'''

class PhysicsInformedNeuralNetworkActions():
    """
    A class used to define the actions of the PINN

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
    SM_model (SM_modelling) : class for creating the synchronous machine model
    machine_params (dict) : parameters of the synchronous machine
    system_params (dict) : parameters of the power system
    modelling_eq (CreateSolver) : class for solving the synchronous machine model
    flag_for_modelling (bool) : flag for using the synchronous machine model
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
    custom_weight_loss_updated(weight_data, weight_dt, weight_pinn, data_loss, dt_loss, pinn_loss, epoch)
        This function updates the weights of the loss functions
    weight_init(module, init_name)
        This function initializes the weights of the neural network model
    test(x_test)
        This function tests the neural network model
    plot(x_train, y_train, var=0)
        This function plots the data for a specific variable
    plot_all(x_train, y_train)
        This function plots all the data in pairs
    plot_all_dt(x_train, y_train)
        This function plots the derivative of all the data in pairs
    forward_nn(time, no_time)
        This function calculates the output of the neural network model, input is given as time and the other input columns
    forward_pass(x_train)
        This function calculates the output of the neural network model, input is given as the one whole tensor


     
    """
    def __init__(self, cfg, modelling_full): # The modelling equations are used, must be predefined, more choices to be added such as dynamic modelling
        self.cfg = cfg
        set_random_seeds(cfg.seed) # set all seeds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

        self.modelling_full = modelling_full
        self.data_loader = DataSampler(cfg)
        self.input_dim = self.data_loader.input_dim # The input dimension is the number of input features
        self.output_dim = self.input_dim-1 # The output dimension is the input dimension minus the time column
        self.model_flag = cfg.model.model_flag  # the model to be used
        self.model = self.define_pinn_model()  # Create an instance of the class Net
        self.weight_init(self.model, cfg.network.weight_init) # Initialize the weights of the Net
        self.criterion = self.custom_loss(cfg.network.loss_criterion) # Define the loss function
        self.criterion_mae = nn.L1Loss() # Define the MAE loss for testing
        self.optimizer = self.custom_optimizer(cfg.network.optimizer, cfg.network.lr) # Define the optimizer
        self.scheduler = self.custom_learning_rate(cfg.network.lr_scheduler) # Define the learning rate scheduler

        # Create an instance of the class xxx_modelling
        if self.cfg.theme == "SM":
            self.SynchronousMachineModels = SynchronousMachineModels(self.cfg)
        elif self.cfg.theme == "GFL":
            self.GridFollowingConverterModels = GridFollowingConverterModels(self.cfg)

        self.model = self.model.to(self.device)
        self.early_stopping = EarlyStopping(patience=cfg.network.early_stopping_patience, verbose=True, delta=cfg.network.early_stopping_min_delta)
        


    def setup_nn(self):
        self.model = self.define_pinn_model()  # Create an instance of the class Net
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

    #Haitian, define pinn
    def define_pinn_model(self):
        """
        This function defines the neural network model
        """
        print("Selected deep learning model: ",self.cfg.network.type)
        if self.cfg.network.type == "KAN": # Static architecture of the neural network
            print(self.input_dim, self.cfg.network.hidden_dim, self.output_dim)
            model = Kalm(self.input_dim, self.cfg.network.hidden_dim, self.output_dim, self.cfg.network.hidden_layers)
            # model.speed()
        elif self.cfg.network.type == "StaticNN": # Static architecture of the neural network
            model = Net(self.input_dim, self.cfg.network.hidden_dim, self.output_dim)
        elif self.cfg.network.type == "DynamicNN" or self.cfg.network.type == "PinnB" or self.cfg.network.type == "PinnA": # Dynamic architecture of the neural network
            model = Network(self.input_dim, self.cfg.network.hidden_dim, self.output_dim, self.cfg.network.hidden_layers)
        elif self.cfg.network.type == "PinnAA": # Dynamic architecture of the neural network with the PinnA architecture for the output
            model = PinnA(self.input_dim, self.cfg.network.hidden_dim, self.output_dim, self.cfg.network.hidden_layers)
        elif self.cfg.network.type == "ResNet":
            num_blocks=2
            num_layers_per_block=2
            model = FullyConnectedResNet(self.input_dim, self.cfg.network.hidden_dim, self.output_dim, num_blocks, num_layers_per_block)
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
        if self.cfg.network.type == "PinnA":
            return no_time + y_pred*time
        if self.cfg.network.type == "PinnB":
            return no_time + y_pred
        if self.cfg.network.type in ["DynamicNN", "PinnAA", "ResNet", "KAN"]:
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
        if self.cfg.network.type == "PinnA":
            return no_time + y_pred*time
        if self.cfg.network.type == "PinnB":
            return no_time + y_pred
        if self.cfg.network.type in ["DynamicNN", "PinnAA", "ResNet","KAN"]:
            return y_pred
        else:
            raise Exception('Enter valid NN type! (zeroth_order or first_order')
        
    def derivative(self, y, t):
        """
        This function calculates the derivative of the model at d_y/d_t
        y1-y0 = dy_dt * (t1-t0), where t1-t0 = dt = 0.1001 for 1000 time intervals
        in order to crosscheck the results, I will calculate the derivative of the output of the model
        and compare it with the derivative of the output of the model
        """
        dy = torch.autograd.grad(y, t, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        return dy


    
    def calculate_autograd22(self, x_train):
        """
        This function calculates the output of the neural network model and the derivative of the output using the derivative function
        """
        time = x_train[:,0].unsqueeze(1) # get the time column

        no_time = x_train[:,1:] # get the input columns
        y = self.forward_nn(time=time, no_time = no_time)
        u = []
        for i in range(y.shape[1]):
            u.append(self.derivative(y[:,i], time))
        u_all = torch.cat(u, 1)
        return y, u_all
    
    def calculate_autograd(self, x_train):
        """
        This function calculates the output of the neural network model and the derivative of the output 
        """
        time = x_train[:,0].unsqueeze(1) # get the time column SOSOSO check if x_train[1:,0] is required 
        no_time = x_train[:,1:] # get the input columns
        y, dy_dt = torch.autograd.functional.jvp( # calculate the jacobian vector product
            func=lambda t: self.forward_nn(time=t, no_time = no_time), inputs=time ,v=torch.ones(time.shape).to(self.device), create_graph=True, retain_graph=True)
        return y, dy_dt
    #Haitian, added for y_processed the "GFL" branch, line 335
    def calculate_from_ode(self, output):
        """
        This function calculates the output(dy/dt) of the synchronous machine model for the given input y
        """
        if self.cfg.modelling_method:
            if self.cfg.theme == "SM":
                y_processed = self.modelling_full.odequation_sm(0, output.split(split_size=1, dim=1))
            if self.cfg.theme == "GFL":
                y_processed = self.modelling_full.odequation_gfl(0, output.split(split_size=1, dim=1))

        else:
            y_processed = self.modelling_full.odequations_v2(0, output.split(split_size=1, dim=1))
        for i in range(len(y_processed)):
            if type(y_processed[i]) == int:
                value = y_processed[i]
                y_processed[i] = torch.tensor(value).repeat(output.shape[0]).unsqueeze(1).to(self.device)
        
        y_processed = torch.cat(y_processed, 1)

        return y_processed



    
    def calculate_point_grad2(self, x_train, y_train):
        """
        This function calculates the pinn loss either for collocation points or for data points
        """
        #autograd
        #y_hat, dy_dt = self.calculate_autograd(x_train)
    
        y_hat, dy_dt = self.calculate_autograd22(x_train) # calculate the output of the model and the derivative of the output
        #ode
        if y_train is None: # collocation points
            y_processed = self.calculate_from_ode(y_hat)
            return dy_dt , y_processed
        else:
            y_processed = self.calculate_from_ode(y_train) # data points
            return y_hat, dy_dt , y_processed
        
    def folder_name_f2(self,cfg):
        weight_data, weight_dt, weight_pinn, weight_pinn_ic = cfg.network.weighting.weights

        self.weight_data = weight_data
        self.weight_dt = weight_dt
        self.weight_pinn = weight_pinn
        self.weight_pinn_ic = weight_pinn_ic

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
    

    

    def update_loss_weights(self, old_weight_data, old_weight_dt, old_weight_pinn, old_weight_pinn_ic, loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch):
        if self.cfg.network.weighting.update_weight_method=="Static":
            return old_weight_data, old_weight_dt, old_weight_pinn, old_weight_pinn_ic
        
        elif self.cfg.network.weighting.update_weight_method=="ReLoBRaLo": #SOS needs fix if this methos is adapted everywhere
            result = self.relobralo_loss(loss_data, loss_dt, loss_pinn, loss_pinn_ic, old_weight_data, old_weight_dt, old_weight_pinn, old_weight_pinn_ic)
            new_weight_data, new_weight_dt, new_weight_pinn, new_weight_pinn_ic = result
            return new_weight_data, new_weight_dt, new_weight_pinn, new_weight_pinn_ic
        
        elif self.cfg.network.weighting.update_weight_method=="Dynamic":
            alpha_max = torch.tensor(0.2)
            epochs_to_tenfold = self.update_weights_freq if self.cfg.network.optimizer == "LBFGS" else self.update_weights_freq*20
            epoch_factor = torch.tensor(10.0 ** (epoch / epochs_to_tenfold))
            new_weight_data = torch.min(alpha_max, self.weight_data * epoch_factor)
            new_weight_dt = torch.min(alpha_max, self.weight_dt * epoch_factor)
            new_weight_pinn = torch.min(alpha_max, self.weight_pinn * epoch_factor)
            new_weight_pinn_ic = torch.min(alpha_max, self.weight_pinn * epoch_factor)
            return new_weight_data, new_weight_dt, new_weight_pinn, new_weight_pinn_ic
        
        elif self.cfg.network.weighting.update_weight_method=="Sam":
            soft_attention_weights = old_weight_data, old_weight_dt, old_weight_pinn, old_weight_pinn_ic
            if epoch % self.update_weights_freq == 0:
                with torch.no_grad():
                    if self.wandb_run is not None:
                        log_data = {f"soft_attention_weights {i}": soft_attention_weights[i] for i in range(4)}
                        log_data.update({f"soft_attention_weights grad{i}": soft_attention_weights.grad[i] for i in range(4)})
                        log_data["epoch"] = epoch
                        self.wandb_run.log(log_data)
                    epsilon = 1e-8
                    soft_attention_weights += self.weight_mask*0.1/(soft_attention_weights.grad + epsilon)
                    soft_attention_weights.grad.zero_()
                new_weight_data, new_weight_dt, new_weight_pinn, new_weight_pinn_ic = soft_attention_weights
                return new_weight_data, new_weight_dt, new_weight_pinn, new_weight_pinn_ic
            else:
                return old_weight_data, old_weight_dt, old_weight_pinn, old_weight_pinn_ic
        
        elif self.cfg.network.update_weight_method=="NTK":
            raise Exception("Not Implemented")
        ########### Add more methods KAN here ###########
        else:
            raise Exception("Weight update method not found")

    def initialize_loss_weights(self, weight_data, weight_dt, weight_pinn, weight_pinn_ic):
        if self.cfg.network.weighting.update_weight_method=="Sam":
            self.soft_attention_weights = torch.nn.Parameter(torch.tensor([weight_data,weight_dt,weight_pinn,weight_pinn_ic], dtype=torch.float32, requires_grad=True))
            self.weight_mask = torch.tensor([weight_data, weight_dt, weight_pinn, weight_pinn_ic], dtype=torch.float32)
            self.weight_mask = torch.where(self.weight_mask == 0, torch.tensor(0.0), torch.tensor(1.0))
            self.update_weights_freq = self.cfg.network.weighting.update_weights_freq

        if self.cfg.network.weighting.update_weight_method=="ReLoBRaLo":
            self.relobralo_loss = ReLoBRaLoLoss()

        if self.cfg.network.weighting.update_weight_method=="Dynamic":
            self.update_weights_freq = self.cfg.network.weighting.update_weights_freq

        self.weight_data = weight_data
        self.weight_dt = weight_dt
        self.weight_pinn = weight_pinn
        self.weight_pinn_ic = weight_pinn_ic
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
    def pinn_train2(self,wandb_run=None):
        """
        This function trains the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
        """
        num_of_skip_data_points = self.cfg.network.num_of_skip_data_points
        num_of_skip_col_points = self.cfg.network.num_of_skip_col_points
        num_of_skip_val_points = self.cfg.network.num_of_skip_val_points
        x_train, y_train, x_train_col, x_train_col_ic, y_train_col_ic, x_val, y_val = self.data_loader.define_train_val_data2(self.cfg.dataset.perc_of_data_points, self.cfg.dataset.perc_of_col_points, num_of_skip_data_points, num_of_skip_col_points, num_of_skip_val_points) # define the training and validation data
        # Create DataLoaders for batch processing
        batch_size = self.cfg.network.batch_size if self.cfg.network.batch_size != "None" else max(len(x_train), len(x_train_col), len(x_train_col_ic))
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size)
        collocation_loader = DataLoader(x_train_col, batch_size=batch_size)
        col_ic_loader = DataLoader(TensorDataset(x_train_col_ic, y_train_col_ic), batch_size=batch_size)

        print("Number of labeled training data:", x_train.shape[0], 
            "Number of collocation points:", x_train_col.shape[0], 
            "Number of collocation points (IC):", x_train_col_ic.shape[0], 
            "Number of validation data:", x_val.shape[0])
        
        folder_name=self.folder_name_f2(self.cfg)
        os.makedirs(os.path.join(self.cfg.dirs.model_dir, folder_name),exist_ok=True)
        self.wandb_run = wandb_run
        
        self.weighting_scheme = PINNWeighting(self.model, self.cfg, self.device, self.output_dim, self.wandb_run)
        # Variable to accumulate the total iteration count
        #total_iteration_count = 0
        # Variable to store the last update iteration
        #last_update_iteration = 0

        print("getting in training")
        for epoch in range(self.cfg.network.num_epochs):
    
            
            self.model.train() # set the model to training mode
            if self.cfg.network.batch_size != "None":
            # Training loop with batches
                for (x_batch, y_batch), x_col_batch, (x_ic_batch, y_ic_batch) in zip(train_loader, collocation_loader, col_ic_loader):
                    
                    # Move data to the correct device
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    x_col_batch = x_col_batch.to(self.device)
                    x_col_ic_batch, y_col_ic_batch = x_ic_batch.to(self.device), y_ic_batch.to(self.device)

                    def closure():
                        #nonlocal iteration_count
                        #iteration_count += 1
                        output, dydt0, ode0 = self.calculate_point_grad2(x_batch, y_batch) # calculate nn output and its gradient for the data points, and the ode solution for the target y_train
                        dydt1, ode1 = self.calculate_point_grad2(x_col_batch, None) # calculate nn output gradient for the collocation points, and the ode solution for the nn output
                        output_col0 = self.forward_pass(x_col_ic_batch) # calculate the nn output for the collocation points with time 0
                        loss_data = self.criterion(output, y_batch) # calculate the data loss
                        
                        loss_dt = [self.criterion(dydt0[:, i], ode0[:, i]) for i in range(dydt0.shape[1])]
                        mean_loss_dt = torch.mean(torch.stack(loss_dt))
                        mean_loss_pinn, loss_pinn = self.calc_adapt_criterion_loss(x_col_batch, dydt1, ode1)
                        loss_pinn = [self.criterion(dydt1[:, i], ode1[:, i]) for i in range(dydt1.shape[1])]
                        mean_loss_pinn = torch.mean(torch.stack(loss_pinn))
                        loss_pinn_ic = self.criterion(output_col0, y_col_ic_batch)
                        loss_total, self.losses = self.weighting_scheme.compute_weighted_loss(loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch) #total_iteration_count + iteration_count
                        self.loss_total = loss_total
                        self.loss_data= loss_data
                        self.loss_dt = mean_loss_dt
                        self.loss_pinn = mean_loss_pinn
                        self.loss_pinn_ic = loss_pinn_ic
                        self.weighting_scheme.log_losses([loss_total, loss_data, mean_loss_dt, mean_loss_pinn, loss_pinn_ic], epoch, ["total_loss","data", "dt", "pinn", "pinn_ic"])

                        self.optimizer.zero_grad() # zero the gradients
                        loss_total.backward() # backpropagate the total weighted loss
                        
                        return loss_total
                
                    self.optimizer.step(closure) # update the weights of the model
            else:
                def closure():
                    output, dydt0, ode0 = self.calculate_point_grad2(x_train, y_train) # calculate nn output and its gradient for the data points, and the ode solution for the target y_train
                    dydt1, ode1 = self.calculate_point_grad2(x_train_col, None)
                    output_col0 = self.forward_pass(x_train_col_ic)
                    loss_data = self.criterion(output, y_train)
                    loss_dt = [self.criterion(dydt0[:, i], ode0[:, i]) for i in range(dydt0.shape[1])]
                    mean_loss_dt = torch.mean(torch.stack(loss_dt))
                    mean_loss_pinn, loss_pinn = self.calc_adapt_criterion_loss(x_train_col, dydt1, ode1)
                    loss_pinn = [self.criterion(dydt1[:, i], ode1[:, i]) for i in range(dydt1.shape[1])]
                    mean_loss_pinn = torch.mean(torch.stack(loss_pinn))
                    loss_pinn_ic = self.criterion(output_col0, y_train_col_ic)
                    loss_total, self.losses = self.weighting_scheme.compute_weighted_loss(loss_data, loss_dt, loss_pinn, loss_pinn_ic, epoch)
                    self.loss_total = loss_total
                    self.loss_data= loss_data
                    self.loss_dt = mean_loss_dt
                    self.loss_pinn = mean_loss_pinn
                    self.loss_pinn_ic = loss_pinn_ic
                    self.weighting_scheme.log_losses([loss_total, loss_data, mean_loss_dt, mean_loss_pinn, loss_pinn_ic], epoch, ["total_loss","data", "dt", "pinn", "pinn_ic"])
                    self.optimizer.zero_grad()
                    loss_total.backward()
                    return loss_total
                
                self.optimizer.step(closure)

            """
            if self.optimizer != "LBFGS":
                self.scheduler.step()
            """
            #total_iteration_count += iteration_count

            # Validation
            self.model.eval()
            
            val_outputs, val_dydt0, val_ode0 = self.calculate_point_grad2(x_val, y_val)
            # add the losses
            loss_val_data = self.criterion(val_outputs, y_val)
            loss_val_physics = [self.criterion(val_dydt0[:, i], val_ode0[:, i]) for i in range(val_dydt0.shape[1])]
            mean_loss_val_physics = torch.mean(torch.stack(loss_val_physics))
                
            val_loss = loss_val_data.item() 
            val_dt_loss = mean_loss_val_physics.item() 
            

            #if total_iteration_count - last_update_iteration > self.cfg.network.weighting.update_weights_freq:
                # update the weights of the loss functions
            #    last_update_iteration = total_iteration_count
            if (epoch + 1) % self.cfg.network.weighting.update_weights_freq == 0:
                if self.cfg.network.weighting.update_weight_method=="Sam":
                    self.weighting_scheme.update_weights(self.losses, epoch)
            
                # log some plots to wandb
                if wandb_run is not None:
                    # self.log_plot(val_outputs, y_val, epoch, wandb_run,x_val)
                    self.log_plot(val_outputs, y_val, epoch, wandb_run, x_val,"validation", 0, 500)
                
            if (epoch + 1 ) % 50 == 0:
                print(f'Epoch [{epoch+1}/{self.cfg.network.num_epochs}], Loss: {self.loss_total.item():.4f}, Loss_data: {self.loss_data.item():.4f}, Loss_dt: {self.loss_dt.item():.4f}, Loss_pinn: {self.loss_pinn.item():.4f} , Loss_pinn_ic : {self.loss_pinn_ic.item():.4f}', val_loss, val_dt_loss)

            # log all the losses for the epoch to wandb 
            save_iteration = 500 if self.cfg.network.optimizer == "LBFGS" else 10000 # 20 iterations within the optimizer ->500*20 = 10000
            if (epoch + 1) % save_iteration == 0:
                
                name = f"{self.cfg.model.model_flag}{self.cfg.network.type}_{self.cfg.time}_{epoch+1}_{self.data_loader.training_shape}_{self.data_loader.training_col_shape}_{self.data_loader.validation_shape}_{self.cfg.dataset.transform_input}_{self.cfg.dataset.transform_output}_{self.weight_data}_{self.weight_dt}_{self.weight_pinn}_{self.weight_pinn_ic}_{self.cfg.network.weighting.update_weight_method}.pth"

                self.save_model(os.path.join(folder_name, name))
            
                if wandb_run is not None:
                    log_data = {
                        "Val_loss": val_loss,
                        "Val_dt_loss": val_dt_loss,
                        "Loss": self.loss_total.item(),
                        "Loss_data": self.loss_data.item(),
                        "Loss_dt": self.loss_dt,
                        "Loss_pinn": self.loss_pinn,
                        "Loss_pinn_ic": self.loss_pinn_ic,
                        "Weight_data": self.weight_data,
                        "Weight_dt": self.weight_dt,
                        "Weight_pinn": self.weight_pinn,
                        "Weight_pinn_ic": self.weight_pinn_ic,
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
        This function tests the neural network model

        Args:
            x_train (torch.Tensor): input data
            y_train (torch.Tensor): output data
            num_epochs (int): number of epochs
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
        if self.cfg.theme == "SM":
            modeling_guide_path = os.path.join(self.cfg.dirs.init_conditions_dir, "modellings_guide_sm.yaml")
            modeling_guide = OmegaConf.load(modeling_guide_path)
        elif self.cfg.theme == "GFL":
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
        gname = f"{type}{blk_idx[0]+1}-{blk_idx[-1]+1}"  # Example ：1-5, 6-10 …
        run.log({f"traj_{gname}": wandb.Image(fig)},commit=False)

        plt.close(fig)
        run.log({}, commit=True)  #  ??


    
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
        if self.cfg.theme == "SM":
            var_name = ["theta","omega(r/s)","E_q(pu)","E_d(pu)"]
        elif self.cfg.theme == "GFL":
            if self.model_flag == "GFL_2nd_order":
                var_name = ["delta", "omega"]
            elif self.model_flag == "GFL_4th_order":
                var_name = ["delta", "delta_omega", "delta_Id", "delta_Id_dt"]
        else:
            raise NotImplementedError
        if run is not None:
            for i in range(y_pred_.shape[1]):
                """
                plt.figure()
                plt.title(f"MAE for variable {var_name[i]} over time")
                """
                mean_mae = np.mean(mae[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), mae[:,i], label=f"Mean MAE: {mean_mae}")
                plt.xlabel("Time(s)")
                plt.ylabel("MAE")
                plt.legend()
                run.log({f"MAE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                plt.figure()
                plt.title(f"RMSE for variable {var_name[i]} over time")
                """
                mean_rmse = np.mean(rmse[:,i])
                """
                plt.plot(unique_values.detach().cpu().numpy(), rmse[:,i], label=f"Mean RMSE: {mean_rmse}")
                plt.xlabel("Time(s)")
                plt.ylabel("RMSE")
                plt.legend()
                run.log({f"RMSE for variable {var_name[i]} over time": wandb.Image(plt)})
                plt.close()
                """
                max_mae = torch.max(mae2[:,i])  # Find the maximum absolute error # calculate the absolute error for each prediction, to find max mae

                #log only mean values
                run.log({f"Mean MAE for variable {self.keys[i]}": mean_mae})
                run.log({f"Mean RMSE for variable {self.keys[i]}": mean_rmse})
                run.log({f"Max MAE for variable {self.keys[i]}": max_mae.item()})
            time = unique_values.detach().cpu().numpy()
            for i in range(y_pred_.shape[1]):
                for j in range(time.shape[0]):
                    #run.log({"Maw ": loss_total.item(), 'epoch': epoch})
                    # log MAE and RMSE for each variable at time  
                
                    run.log({f"MAE for variable {self.keys[i]}": mae[j,i], "Time": time[j]})
                    run.log({f"RMSE for variable {self.keys[i]}": rmse[j,i], "Time": time[j]})

        max_mae = torch.max(mae2)  # Find the maximum absolute error
        run.log({"Test Max AE": max_mae.item()})
        #save the mae and rmse
        full_path = os.path.join(self.cfg.dirs.model_dir, self.final_name)
        np.save(full_path+"_mae.npy", mae)
        np.save(full_path+"_rmse.npy", rmse)

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