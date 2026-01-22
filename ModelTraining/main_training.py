import seResNet   # SE_ResNet1D here
import utils
import torch
from torch.utils.data import TensorDataset, DataLoader

data_base_dir = "DATADIRECTORY"      # Replace with your actual data directory path
model_save_path = "put_your_model_save_path_here"  # <- set this properly

# === Load ALL data ===
all_inputs, all_targets, all_pat_ids = utils.load_all_patients(data_base_dir, 1280)

# === Build a single DataLoader over the entire dataset ===
# Make sure tensors have correct dtype
if not isinstance(all_inputs, torch.Tensor):
    all_inputs_tensor = torch.from_numpy(all_inputs).float()
else:
    all_inputs_tensor = all_inputs

if not isinstance(all_targets, torch.Tensor):
    all_targets_tensor = torch.from_numpy(all_targets).float()
else:
    all_targets_tensor = all_targets

idx_tensor = torch.arange(all_inputs_tensor.shape[0])

batch_size = 32
full_dataset = TensorDataset(all_inputs_tensor, all_targets_tensor, idx_tensor)
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

# === SE-ResNet1D model definition ===
num_blocks   = 5
channels     = [32, 64, 128, 256, 512]
kernel_sizes = [9, 7, 5, 3, 3]
use_residual = True
dropout_rate = 0.1
reduction    = 4

model = seResNet.SE_ResNet1D(
    input_channels=2,
    num_blocks=num_blocks,
    channels=channels,
    kernel_sizes=kernel_sizes,
    reduction=reduction,
    use_residual=use_residual,
    dropout_rate=dropout_rate,
)

# === Training parameters ===
loss_function = utils.RRMSELoss()
lr = 0.001
num_epochs = 61

model_id = 0
model_name = f"model{model_id}"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# === Train on FULL dataset and save weights ===
model = utils.train_model_full(
    model=model,
    model_id=model_name,
    train_loader=train_loader,
    loss_function=loss_function,
    device=device,
    num_epochs=num_epochs,
    lr=lr,
    model_save_path=model_save_path
)
