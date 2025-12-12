import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def run_pretrain(script_name, npz_path, save_dir, epochs, batch_size, lr):
    os.makedirs(save_dir, exist_ok=True)
    base_dir = os.path.dirname(__file__)
    script_path = os.path.join(base_dir, script_name)

    cmd = (
        f"{sys.executable} {script_path} "
        f"--npz {npz_path} --epochs {epochs} "
        f"--batch-size {batch_size} --lr {lr} --save-dir {save_dir}"
    )
    print("Running:", cmd)
    os.system(cmd)


def plot_log(save_dir, log_name, title):
    log_file = os.path.join(save_dir, log_name)
    if not os.path.exists(log_file):
        print(f"No log file {log_name} in {save_dir} for visualization.")
        return

    log = np.load(log_file, allow_pickle=True).item()
    train = log["train"]
    val = log["val"]

    plt.figure()
    plt.plot(train, label="Train MSE")
    plt.plot(val, label="Val MSE")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, title.replace(" ", "_").lower() + ".png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss plot to {out_path}")


def main():
    base_dir = os.path.dirname(__file__)
    npz_path = os.path.abspath(os.path.join(base_dir, "pretraining_actor_critic_input.npz"))

    # ONE shared dir for both:
    save_dir = os.path.join(base_dir, "pretrained_sbto")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using NPZ: {npz_path}")
    print(f"Saving actor & critic weights into: {save_dir}")

    # Actor pretraining
    run_pretrain(
        script_name="pretrain_actor.py",
        npz_path=npz_path,
        save_dir=save_dir,
        epochs=10,
        batch_size=1024,
        lr=1e-3,
    )

    # Critic pretraining
    run_pretrain(
        script_name="pretrain_critic.py",
        npz_path=npz_path,
        save_dir=save_dir,
        epochs=10,
        batch_size=1024,
        lr=1e-3,
    )

    # Plots 
    plot_log(save_dir, "training_log_actor.npy", "Actor Pretraining Loss")
    plot_log(save_dir, "training_log_critic.npy", "Critic Pretraining Loss")


if __name__ == "__main__":
    main()