import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from rsl_rl.networks import MLP, EmpiricalNormalization


# Dataset

class SbtoNpzDataset(Dataset):
    """
    Builds all 8 MJLAB policy inputs from a rollout npz with shape (N_traj, T, ·).

    Input at time t  ->  target = joint positions at time t+1.
    """
    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)

        # Raw arrays with 3 dims: (N_traj, T, ·)
        raw_act_pos = data["actuator_pos"].astype(np.float32)           # (N, T, A)
        raw_act_vel = data["actuator_vel"].astype(np.float32)           # (N, T, A)
        raw_base_linang = data["base_linvel_angvel"].astype(np.float32) # (N, T, 6)

        N, T, A = raw_act_pos.shape
        if T < 2:
            raise ValueError("Need at least 2 timesteps to build t -> t+1 pairs")

        self.N_traj = N
        self.T_traj = T
        self.A = A

        #  base linear and angular velocities in 3D 
        raw_base_lin = raw_base_linang[..., :3]  # (N, T, 3)
        raw_base_ang = raw_base_linang[..., 3:]  # (N, T, 3)

        
        # last_action[t] = act_pos[t-1], last_action[0] = 0
        raw_last_action = np.zeros_like(raw_act_pos)   # (N, T, A)
        raw_last_action[:, 1:, :] = raw_act_pos[:, :-1, :]

        # anchor terms 
        raw_anchor_pos_b = np.zeros((N, T, 3), dtype=np.float32)       # (N, T, 3)
        raw_anchor_ori_b = np.tile(
            np.array([1, 0, 0, 0, 1, 0], dtype=np.float32),
            (N, T, 1)
        )  # (N, T, 6)

        # Effective timesteps 
        T_eff = T - 1

        # Inputs at time t
        obs_act_pos      = raw_act_pos[:, :-1, :]        # (N, T-1, A)
        obs_act_vel      = raw_act_vel[:, :-1, :]        # (N, T-1, A)
        obs_base_lin     = raw_base_lin[:, :-1, :]       # (N, T-1, 3)
        obs_base_ang     = raw_base_ang[:, :-1, :]       # (N, T-1, 3)
        obs_last_action  = raw_last_action[:, :-1, :]    # (N, T-1, A)
        obs_anchor_pos_b = raw_anchor_pos_b[:, :-1, :]   # (N, T-1, 3)
        obs_anchor_ori_b = raw_anchor_ori_b[:, :-1, :]   # (N, T-1, 6)

        # Targets at time t+1  
        target_act_pos = raw_act_pos[:, 1:, :]           # (N, T-1, A)

        #flatten to 2D
        self.act_pos   = obs_act_pos.reshape(N * T_eff, A)        # (N*(T-1), A)
        self.act_vel   = obs_act_vel.reshape(N * T_eff, A)        # (N*(T-1), A)
        self.base_lin  = obs_base_lin.reshape(N * T_eff, 3)       # (N*(T-1), 3)
        self.base_ang  = obs_base_ang.reshape(N * T_eff, 3)       # (N*(T-1), 3)
        self.last_action   = obs_last_action.reshape(N * T_eff, A)
        self.anchor_pos_b  = obs_anchor_pos_b.reshape(N * T_eff, 3)
        self.anchor_ori_b  = obs_anchor_ori_b.reshape(N * T_eff, 6)

        # generated_commands 
        self.commands = np.concatenate([self.act_pos, self.act_vel], axis=-1)  # (N*(T-1), 2A)

        # Targets: joint_pos at t+1
        self.targets = target_act_pos.reshape(N * T_eff, A)   # (N*(T-1), A)
        self.num_samples = N * T_eff

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = {
            "command":        torch.from_numpy(self.commands[idx]),      # (2A,)
            "anchor_pos_b":   torch.from_numpy(self.anchor_pos_b[idx]),  # (3,)
            "anchor_ori_b":   torch.from_numpy(self.anchor_ori_b[idx]),  # (6,)
            "base_lin_vel":   torch.from_numpy(self.base_lin[idx]),      # (3,)
            "base_ang_vel":   torch.from_numpy(self.base_ang[idx]),      # (3,)
            "joint_pos":      torch.from_numpy(self.act_pos[idx]),       # (A,)  at time t
            "joint_vel":      torch.from_numpy(self.act_vel[idx]),       # (A,)  at time t
            "last_action":    torch.from_numpy(self.last_action[idx]),   # (A,)
        }
        target = torch.from_numpy(self.targets[idx])                      # (A,) = joint_pos at t+1
        return obs, target



def collate_to_actor_obs(batch):
    """
    Concatenate in MJLAB order:
    command, anchor_pos_b, anchor_ori_b, base_lin_vel, base_ang_vel, joint_pos, joint_vel, last_action
    """
    cmd   = torch.stack([b[0]["command"]      for b in batch], dim=0)
    posb  = torch.stack([b[0]["anchor_pos_b"] for b in batch], dim=0)
    orib  = torch.stack([b[0]["anchor_ori_b"] for b in batch], dim=0)
    blin  = torch.stack([b[0]["base_lin_vel"] for b in batch], dim=0)
    bang  = torch.stack([b[0]["base_ang_vel"] for b in batch], dim=0)
    jpos  = torch.stack([b[0]["joint_pos"]    for b in batch], dim=0)
    jvel  = torch.stack([b[0]["joint_vel"]    for b in batch], dim=0)
    last  = torch.stack([b[0]["last_action"]  for b in batch], dim=0)

    actor_obs = torch.cat([cmd, posb, orib, blin, bang, jpos, jvel, last], dim=-1)
    targets   = torch.stack([b[1] for b in batch], dim=0)
    return actor_obs, targets



class ActorMLP(nn.Module):
    """
    Matches MJLAB actor head: 3x256 ELU MLP -> num_actions outputs.
    Uses EmpiricalNormalization on inputs to mirror MJLAB behavior.
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden=(256, 256, 256), activation="elu"):
        super().__init__()
        self.norm = EmpiricalNormalization(obs_dim) #normalize inputs and keep updating during training
        self.mlp = MLP(obs_dim, num_actions, hidden, activation)

    def forward(self, x):
        x = self.norm(x) #normalize inputs
        return self.mlp(x)


# Train / Validation

def train_epoch(model, loader, optim, device):
    
    model.train()
    total = 0.0
    n = 0
    for actor_obs, targets in loader:
        actor_obs = actor_obs.to(device)
        targets = targets.to(device)
        # Update normalizer 
        model.norm.update(actor_obs)

        pred = model(actor_obs)  # (B, A)
        loss = nn.functional.mse_loss(pred, targets) 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += loss.item() * actor_obs.size(0) #accumulate total loss over all batches
        n += actor_obs.size(0)
    return total / max(1, n) #average loss across all samples


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for actor_obs, targets in loader:
        actor_obs = actor_obs.to(device)
        targets = targets.to(device)
        actor_obs = model.norm(actor_obs)
        pred = model.mlp(actor_obs)
        loss = nn.functional.mse_loss(pred, targets)
        total += loss.item() * actor_obs.size(0)
        n += actor_obs.size(0)
    return total / max(1, n)


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="Path to time_x_u_traj_rl_format.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", type=str, default="./pretrained_actor")
    args = ap.parse_args()

    torch.manual_seed(args.seed)  #to fix both data splitting and weight initialization
    np.random.seed(args.seed)

    ds = SbtoNpzDataset(args.npz)
    A = ds.act_pos.shape[1]
    obs_dim = (2 * A) + 3 + 6 + 3 + 3 + A + A + A  #computed actor obs dim
    # Split
    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    split_idx = int(0.8 * len(ds))
    train_ds = torch.utils.data.Subset(ds, range(0, split_idx))
    val_ds = torch.utils.data.Subset(ds, range(split_idx, len(ds)))

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_to_actor_obs, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_to_actor_obs, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ActorMLP(obs_dim=obs_dim, num_actions=A, hidden=(256, 256, 256), activation="elu").to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    os.makedirs(args.save_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, device)
        va = eval_epoch(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_mse={tr:.6f}  val_mse={va:.6f}")
        train_losses.append(tr)
        val_losses.append(va)

        # save best
        if va < best_val:
            best_val = va
            save_path = os.path.join(args.save_dir, "actor_from_sbto_best.pth")
            torch.save({
                "actor_state_dict": model.state_dict(),     # includes norm + mlp
                "obs_dim": obs_dim,
                "num_actions": A,
                "hidden": (256, 256, 256),
                "activation": "elu",
                "val_mse": best_val,
            }, save_path)
            print(f"  -> saved: {save_path}")

    # also save final
    final_path = os.path.join(args.save_dir, "actor_from_sbto_last.pth")
    torch.save({
        "actor_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "num_actions": A,
        "hidden": (256, 256, 256),
        "activation": "elu",
        "val_mse": best_val,
    }, final_path)

    # Save training logs for visualization
    np.save(
        os.path.join(args.save_dir, "training_log.npy"),
        {"train": train_losses, "val": val_losses},
    )
    print(f"Saved training log to {args.save_dir}")
    print(f"Saved training log to {args.save_dir}")
    print(f"  -> saved: {final_path}")



if __name__ == "__main__":
    main()