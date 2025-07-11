import matplotlib.pyplot as plt
import omegaconf as Omegaconf
import P

dataset_path = PowerPINN/data/GFL_2nd_order/dataset_v1.pkl #os.path.join(self.cfg.dirs.dataset_dir, name) # Define the path to the dataset
print("Loading data from: ", dataset_path)
with open(dataset_path, 'rb') as f:
    sol = pickle.load(f)