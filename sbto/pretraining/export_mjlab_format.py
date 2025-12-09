import argparse
import os
import torch

"""
Merges SBTO-pretrained nets with a full MJLAB checkpoint.

- sbto_ckpt  provides:
    model_state_dict = {actor.*, actor_obs_normalizer.*,
                        critic.*, critic_obs_normalizer.*, std}
  → all actor + critic weights and normalizers come from sbto pretraining.

- base_ckpt (e.g. model_500.pt) provides:
    optimizer_state_dict and other metadata
  →  keep optimizer/config


 the final output file uses generated SBTO actor/critic (no RL weights left),
but still has a valid optimizer_state_dict etc, for MJLAB to load.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sbto-ckpt",
        type=str,
        default="sbto/pretraining/pretrained_sbto_results/model_sbto_pretrained.pt",
        help="SBTO checkpoint with model_state_dict (actor+critic+norms+std)",
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="sbto/pretraining/pretrained_sbto_results/model_500_sbto_actor.pt",
        help="Original MJLAB checkpoint to steal optimizer_state_dict etc. from",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="model_sbto_full.pt",
        help="Output MJLAB-style checkpoint file",
    )
    args = parser.parse_args()

    sbto_path = args.sbto_ckpt
    base_path = args.base_ckpt
    script_dir = os.path.dirname(__file__)
    forced_dir = os.path.abspath(os.path.join(script_dir, "..", "pretraining"))
    os.makedirs(forced_dir, exist_ok=True)

    out_path = os.path.join(forced_dir, args.out)

    if not os.path.exists(sbto_path):
        raise FileNotFoundError(f"SBTO checkpoint not found: {sbto_path}")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base checkpoint not found: {base_path}")

    print(f"Loading SBTO checkpoint from: {sbto_path}")
    sbto_ckpt = torch.load(sbto_path, map_location="cpu")
    if "model_state_dict" not in sbto_ckpt:
        raise KeyError("SBTO checkpoint has no key 'model_state_dict'.")
    sbto_model_sd = sbto_ckpt["model_state_dict"]

    print(f"Loading base MJLAB checkpoint from: {base_path}")
    base_ckpt = torch.load(base_path, map_location="cpu")
    if "model_state_dict" not in base_ckpt:
        raise KeyError("Base checkpoint has no key 'model_state_dict'.")

    # Optional sanity check
    print("Replacing base model_state_dict with SBTO model_state_dict")
    print(f"  SBTO keys: {len(sbto_model_sd)} | Base keys: {len(base_ckpt['model_state_dict'])}")

    base_ckpt["model_state_dict"] = sbto_model_sd

    torch.save(base_ckpt, out_path)
    print(f"Saved merged checkpoint to: {out_path}")


if __name__ == "__main__":
    main()