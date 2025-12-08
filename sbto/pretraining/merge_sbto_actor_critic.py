import os
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--actor", type=str, default="actor_from_sbto_best.pth")
    p.add_argument("--critic", type=str, default="critic_from_sbto_best.pth")
    p.add_argument("--out", type=str, default="model_sbto_pretrained.pt")
    args = p.parse_args()

    actor_path = os.path.join(args.save_dir, args.actor)
    critic_path = os.path.join(args.save_dir, args.critic)
    out_path = os.path.join(args.save_dir, args.out)

    actor_ckpt = torch.load(actor_path, map_location="cpu")
    critic_ckpt = torch.load(critic_path, map_location="cpu")

    actor_sd = actor_ckpt["actor_state_dict"]
    critic_sd = critic_ckpt["critic_state_dict"]

    model_sd = {}

    for k, v in actor_sd.items():
        if k.startswith("norm."):
            new_k = "actor_obs_normalizer." + k[len("norm.") :]
        elif k.startswith("mlp."):
            new_k = "actor." + k[len("mlp.") :]
        else:
            continue
        model_sd[new_k] = v

    for k, v in critic_sd.items():
        if k.startswith("norm."):
            new_k = "critic_obs_normalizer." + k[len("norm.") :]
        elif k.startswith("mlp."):
            new_k = "critic." + k[len("mlp.") :]
        else:
            continue
        model_sd[new_k] = v

    action_dim = model_sd["actor.6.bias"].shape[0]
    model_sd["std"] = torch.zeros(action_dim)

    ckpt = {"model_state_dict": model_sd}
    torch.save(ckpt, out_path)
    print(f"Saved merged model to {out_path}")


if __name__ == "__main__":
    main()





"""

HOW TO RUN:

cd sbto/pretraining   
python merge_sbto_actor_critic.py \
  --save-dir ./pretrained_sbto \
  --out model_sbto_pretrained.pt
  
  """
