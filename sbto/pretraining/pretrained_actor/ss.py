import torch
from rsl_rl.networks import MLP, EmpiricalNormalization

# Path to your trained model file
path = "/Users/mustaphadaly/Desktop/sbto/sbto/pretraining/pretrained_actor/actor_from_sbto_best.pth"

# Load checkpoint
ckpt = torch.load(path, map_location="cpu")

# Extract info
print("Keys in checkpoint:", ckpt.keys())
print("Validation MSE:", ckpt["val_mse"])

# Rebuild model using saved parameters
model = torch.nn.Sequential(
    EmpiricalNormalization(ckpt["obs_dim"]),
    MLP(ckpt["obs_dim"], ckpt["num_actions"], ckpt["hidden"], ckpt["activation"])
)

# Load weights
model.load_state_dict(ckpt["actor_state_dict"], strict=False)

# Print structure
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")