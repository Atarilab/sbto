import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

def main(): 
    #Training
    base_dir = os.path.dirname(__file__)
    npz_path = os.path.abspath(os.path.join(base_dir, "../data/rollout_time_x_u_obs_traj_scene_29dof_rl_format.npz"))
    save_dir = os.path.join(base_dir, "pretrained_actor")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Running supervised pretraining on: {npz_path}")
    os.system(
        f"{sys.executable} {os.path.join(base_dir, 'pretrain_actor_from_npz.py')} "
        f"--npz {npz_path} --epochs 10 --batch-size 1024 --lr 1e-3 --save-dir {save_dir}"
    )

    print("Results saved to:", save_dir)

    # Visualization 
    log_file = os.path.join(save_dir, "training_log.npy")

    if os.path.exists(log_file):
        log = np.load(log_file, allow_pickle=True).item()
        train = log["train"]
        val = log["val"]

        plt.plot(train, label="Train MSE", linewidth=2)
        plt.plot(val, label="Validation MSE", linewidth=2)
        plt.title("Pretraining Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(" No log file found for visualization.")

    # Network info
    
    print("\nRunning network info summary...")
    exec(open("/Users/mustaphadaly/Desktop/sbto/sbto/pretraining/pretrained_actor/network_architecture.py").read())

if __name__ == "__main__":
    main()