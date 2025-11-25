import torch
import numpy as np
import os
from pretrain_actor_from_npz import ActorMLP

def main():
    # Paths
    save_dir = "sbto/pretraining/pretrained_actor"
    ckpt_path = os.path.join(save_dir, "actor_from_sbto_best.pth")
    onnx_out = os.path.join(save_dir, "actor_from_sbto.onnx")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    obs_dim     = ckpt["obs_dim"]
    num_actions = ckpt["num_actions"]
    hidden      = ckpt["hidden"]
    activation  = ckpt["activation"]

    # Rebuild model architecture
    model = ActorMLP(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden=hidden,
        activation=activation
    ).cpu()

    # Load weights
    model.load_state_dict(ckpt["actor_state_dict"])
    model.eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_out,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=13,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}
    )

    print(f"Exported ONNX to: {onnx_out}")

if __name__ == "__main__":
    main()