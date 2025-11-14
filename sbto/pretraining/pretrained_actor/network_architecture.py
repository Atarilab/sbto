import os
import torch

def main():
    path = "/Users/mustaphadaly/Desktop/sbto/sbto/pretraining/pretrained_actor/actor_from_sbto_last.pth"
    ckpt = torch.load(path, map_location="cpu")

    # Extract model information
    obs_dim = ckpt["obs_dim"]
    num_actions = ckpt["num_actions"]
    hidden = ckpt["hidden"]
    activation = ckpt["activation"]
    val_mse = ckpt["val_mse"]

    # Compute total parameters
    total_params = sum(p.numel() for p in ckpt["actor_state_dict"].values())

    # === Save to text file ===
    output_path = os.path.join(os.path.dirname(path), "network_info.txt")
    with open(output_path, "w") as f:
        f.write("=== NETWORK SUMMARY ===\n")
        f.write(f"Input size (obs_dim): {obs_dim}\n")
        f.write(f"Output size (num_actions): {num_actions}\n")
        f.write(f"Hidden layers: {hidden}\n")
        f.write(f"Activation: {activation}\n")
        f.write(f"Validation MSE: {val_mse:.6f}\n\n")
        f.write("=== TRAINING PARAMETERS ===\n")
        f.write("Epochs: 30\nBatch size: 1024\nLearning rate: 1e-3\n\n")
        f.write("=== MODEL STATS ===\n")
        f.write(f"Total parameters: {total_params}\n")

    print(f"\nSaved network info to: {output_path}")

if __name__ == "__main__":
    main()