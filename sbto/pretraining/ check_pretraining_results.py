"""
check_pretraining_results.py
Utility script to visualize and validate pretraining results.
"""

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from typing import List
import numpy as np

# --- Load training logs (replace with your stored values if needed) ---
train_losses: List[float] = []
val_losses: List[float] = []

# val_losses = np.loadtxt("val_losses.txt").tolist()

# --- Plot training vs validation MSE ---
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train MSE", linewidth=2)
plt.plot(val_losses, label="Val MSE", linewidth=2)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.title("Supervised Pretraining MSE", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pretraining_mse_curve.png")
plt.show()

# --- Load model for sanity check ---
from sbto.pretraining.mjlab_actor import ActorNetwork  # adjust if class has different name

model_path = "sbto/pretraining/pretrained_actor/actor_from_sbto_best.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ActorNetwork(58, 29)  # use correct input/output dims from training
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- Load dataset for quick check ---
npz_path = "sbto/data/time_x_u_traj_rl_format.npz"
data = np.load(npz_path)
X = np.hstack([data["actuator_pos"], data["actuator_vel"], data["base_xyz_quat"], data["base_linvel_angvel"]])
Y = data["actuator_pos"]

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

with torch.no_grad():
    Y_pred = model(X_tensor[:5]).cpu().numpy()
    Y_true = Y_tensor[:5].cpu().numpy()

print("\n--- Sanity check: predictions vs targets (first 3 samples) ---")
for i in range(3):
    print(f"Sample {i+1}:")
    print(f"Pred:   {Y_pred[i][:5]} ...")
    print(f"Target: {Y_true[i][:5]} ...\n")

print("Plot saved as pretraining_mse_curve.png ")