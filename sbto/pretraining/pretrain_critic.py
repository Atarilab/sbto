import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rsl_rl.networks import MLP, EmpiricalNormalization



class SbtoCriticDataset(Dataset):
    """
     MJLAB critic inputs from pretraining NPZ.
   
    """
    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)

        joint_pos            = data["joint_pos"].astype(np.float32)            # (N, T, 29)
        joint_vel            = data["joint_vel"].astype(np.float32)            # (N, T, 29)
        motion_anchor_pos_b  = data["motion_anchor_pos_b"].astype(np.float32)  # (N, T, 3)
        motion_anchor_ori_b  = data["motion_anchor_ori_b"].astype(np.float32)  # (N, T, 6)
        body_pos             = data["body_pos"].astype(np.float32)             # (N, T, 42)
        body_ori             = data["body_ori"].astype(np.float32)             # (N, T, 84)
        base_lin_vel         = data["base_lin_vel"].astype(np.float32)         # (N, T, 3)
        base_ang_vel         = data["base_ang_vel"].astype(np.float32)         # (N, T, 3)
        actions              = data["actions"].astype(np.float32)              # (N, T, 29)
        object_global_pos    = data["object_global_pos"].astype(np.float32)    # (N, T, 30)
        object_global_ori    = data["object_global_ori"].astype(np.float32)    # (N, T, 60)
        object_lin_vel_w     = data["object_lin_vel_w"].astype(np.float32)     # (N, T, 30)
        object_ang_vel_w     = data["object_ang_vel_w"].astype(np.float32)     # (N, T, 30)
        object_pos_error     = data["object_pos_error"].astype(np.float32)     # (N, T, 30)
        object_ori_error     = data["object_ori_error"].astype(np.float32)     # (N, T, 60)
        
        N, T, A = joint_pos.shape
        self.N_traj = N
        self.T_traj = T
        self.A = A

        command = np.concatenate([joint_pos, joint_vel], axis=-1)  
        
        self.command             = command.reshape(N * T, 58)
        self.motion_anchor_pos_b = motion_anchor_pos_b.reshape(N * T, 3)
        self.motion_anchor_ori_b = motion_anchor_ori_b.reshape(N * T, 6)
        self.body_pos            = body_pos.reshape(N * T, 42)
        self.body_ori            = body_ori.reshape(N * T, 84)
        self.base_lin_vel        = base_lin_vel.reshape(N * T, 3)
        self.base_ang_vel        = base_ang_vel.reshape(N * T, 3)
        self.joint_pos           = joint_pos.reshape(N * T, A)
        self.joint_vel           = joint_vel.reshape(N * T, A)
        self.actions             = actions.reshape(N * T, A)
        self.object_global_pos   = object_global_pos.reshape(N * T, 30)
        self.object_global_ori   = object_global_ori.reshape(N * T, 60)
        self.object_lin_vel_w    = object_lin_vel_w.reshape(N * T, 30)
        self.object_ang_vel_w    = object_ang_vel_w.reshape(N * T, 30)
        self.object_pos_error    = object_pos_error.reshape(N * T, 30)
        self.object_ori_error    = object_ori_error.reshape(N * T, 60)

        self.targets = self._compute_targets_from_errors(
            object_pos_error, object_ori_error
        )   

        # critic obs 
        self.obs_dim = (
            self.command.shape[1]             # 58
            + self.motion_anchor_pos_b.shape[1]  # 3
            + self.motion_anchor_ori_b.shape[1]  # 6
            + self.body_pos.shape[1]          # 42
            + self.body_ori.shape[1]          # 84
            + self.base_lin_vel.shape[1]      # 3
            + self.base_ang_vel.shape[1]      # 3
            + self.joint_pos.shape[1]         # 29
            + self.joint_vel.shape[1]         # 29
            + self.actions.shape[1]           # 29
            + self.object_global_pos.shape[1] # 30
            + self.object_global_ori.shape[1] # 60
            + self.object_lin_vel_w.shape[1]  # 30
            + self.object_ang_vel_w.shape[1]  # 30
            + self.object_pos_error.shape[1]  # 30
            + self.object_ori_error.shape[1]  # 60
        )

        self.num_samples = N * T
    def _compute_targets_from_errors(self, object_pos_error, object_ori_error):
        """
        Compute critic targets using the same reward logic as MJLab's
        object_global_pos + object_global_ori terms:
    
        r     = w_pos * r_pos + w_ori * r_ori

        """

        N, T, _ = object_pos_error.shape

        # same std and weights as env_cfg (from mjlab2):
        std_pos = 0.25
        std_ori = 0.3
        w_pos = 1.0
        w_ori = 0.8
        gamma = 0.99 


        pos_err_sq = np.sum(object_pos_error**2, axis=-1)  # (N, T)
        ori_err_sq = np.sum(object_ori_error**2, axis=-1)  # (N, T)

        r_pos = np.exp(-pos_err_sq / (std_pos**2))         # (N, T)
        r_ori = np.exp(-ori_err_sq / (std_ori**2))         # (N, T)

        r_step = w_pos * r_pos + w_ori * r_ori             # (N, T)

        # discounted returns per step
        returns = np.zeros_like(r_step, dtype=np.float32)  # (N, T)
        returns[:, -1] = r_step[:, -1]
        for t in range(T - 2, -1, -1):
            returns[:, t] = r_step[:, t] + gamma * returns[:, t + 1]

        # flatten to (N*T, 1) as training targets
        return returns.reshape(N * T, 1).astype(np.float32)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = {
            "command":             torch.from_numpy(self.command[idx]),
            "motion_anchor_pos_b": torch.from_numpy(self.motion_anchor_pos_b[idx]),
            "motion_anchor_ori_b": torch.from_numpy(self.motion_anchor_ori_b[idx]),
            "body_pos":            torch.from_numpy(self.body_pos[idx]),
            "body_ori":            torch.from_numpy(self.body_ori[idx]),
            "base_lin_vel":        torch.from_numpy(self.base_lin_vel[idx]),
            "base_ang_vel":        torch.from_numpy(self.base_ang_vel[idx]),
            "joint_pos":           torch.from_numpy(self.joint_pos[idx]),
            "joint_vel":           torch.from_numpy(self.joint_vel[idx]),
            "actions":             torch.from_numpy(self.actions[idx]),
            "object_global_pos":   torch.from_numpy(self.object_global_pos[idx]),
            "object_global_ori":   torch.from_numpy(self.object_global_ori[idx]),
            "object_lin_vel_w":    torch.from_numpy(self.object_lin_vel_w[idx]),
            "object_ang_vel_w":    torch.from_numpy(self.object_ang_vel_w[idx]),
            "object_pos_error":    torch.from_numpy(self.object_pos_error[idx]),
            "object_ori_error":    torch.from_numpy(self.object_ori_error[idx]),
        }
        target = torch.from_numpy(self.targets[idx])
        return obs, target


def collate_to_critic_obs(batch):
    # exactly in mjlab critic order
    cmd   = torch.stack([b[0]["command"]             for b in batch], dim=0)
    mpos  = torch.stack([b[0]["motion_anchor_pos_b"] for b in batch], dim=0)
    mori  = torch.stack([b[0]["motion_anchor_ori_b"] for b in batch], dim=0)
    bpos  = torch.stack([b[0]["body_pos"]            for b in batch], dim=0)
    bori  = torch.stack([b[0]["body_ori"]            for b in batch], dim=0)
    blin  = torch.stack([b[0]["base_lin_vel"]        for b in batch], dim=0)
    bang  = torch.stack([b[0]["base_ang_vel"]        for b in batch], dim=0)
    jpos  = torch.stack([b[0]["joint_pos"]           for b in batch], dim=0)
    jvel  = torch.stack([b[0]["joint_vel"]           for b in batch], dim=0)
    acts  = torch.stack([b[0]["actions"]             for b in batch], dim=0)
    ogpos = torch.stack([b[0]["object_global_pos"]   for b in batch], dim=0)
    ogori = torch.stack([b[0]["object_global_ori"]   for b in batch], dim=0)
    olin  = torch.stack([b[0]["object_lin_vel_w"]    for b in batch], dim=0)
    oang  = torch.stack([b[0]["object_ang_vel_w"]    for b in batch], dim=0)
    opos  = torch.stack([b[0]["object_pos_error"]    for b in batch], dim=0)
    oori  = torch.stack([b[0]["object_ori_error"]    for b in batch], dim=0)

    critic_obs = torch.cat(
        [cmd, mpos, mori, bpos, bori, blin, bang,
         jpos, jvel, acts,
         ogpos, ogori, olin, oang, opos, oori],
        dim=-1,
    )  # (B, 526)

    targets = torch.stack([b[1] for b in batch], dim=0)  # (B, 1)
    return critic_obs, targets


class CriticMLP(nn.Module):
    """Critic MLP: (512, 256, 128) ELU + EmpiricalNormalization, like mjlab config."""
    def __init__(self, obs_dim: int, hidden=(512, 256, 128), activation="elu"):
        super().__init__()
        self.norm = EmpiricalNormalization(obs_dim)
        self.mlp = MLP(obs_dim, 1, hidden, activation)

    def forward(self, x):
        x = self.norm(x)
        return self.mlp(x)

#training
def train_epoch(model, loader, optim, device):
    model.train()
    total = 0.0
    n = 0
    for critic_obs, targets in loader:
        critic_obs = critic_obs.to(device)
        targets = targets.to(device)
        model.norm.update(critic_obs)
        pred = model(critic_obs)           # (B, 1)
        loss = nn.functional.mse_loss(pred, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        total += loss.item() * critic_obs.size(0)
        n += critic_obs.size(0)
    return total / max(1, n)

# evaluation
@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for critic_obs, targets in loader:
        critic_obs = critic_obs.to(device)
        targets = targets.to(device)
        critic_obs = model.norm(critic_obs)
        pred = model.mlp(critic_obs)
        loss = nn.functional.mse_loss(pred, targets)
        total += loss.item() * critic_obs.size(0)
        n += critic_obs.size(0)
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", type=str, default="./pretrained_critic")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = SbtoCriticDataset(args.npz)
    obs_dim = ds.obs_dim

    train_size = int(0.8 * len(ds)) 
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_to_critic_obs,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_to_critic_obs,
        pin_memory=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CriticMLP(
        obs_dim=obs_dim,
        hidden=(512, 256, 128),
        activation="elu",
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = math.inf
    os.makedirs(args.save_dir, exist_ok=True)
    train_hist = []
    val_hist = []
    
    # training loop
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, device)
        va = eval_epoch(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_mse={tr:.6f}  val_mse={va:.6f}")
        train_hist.append(tr)
        val_hist.append(va)


        if va < best_val:
            best_val = va
            save_path = os.path.join(args.save_dir, "critic_from_sbto_best.pth")
            torch.save(
                {
                    "critic_state_dict": model.state_dict(),  # includes norm + mlp
                    "obs_dim": obs_dim,
                    "hidden": (512, 256, 128),
                    "activation": "elu",
                    "val_mse": best_val,
                },
                save_path,
            )
            print(f"  -> saved: {save_path}")

    final_path = os.path.join(args.save_dir, "critic_from_sbto_last.pth")
    torch.save(
        {
            "critic_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "hidden": (512, 256, 128),
            "activation": "elu",
            "val_mse": best_val,
        },
        final_path,
    )
    log_path = os.path.join(args.save_dir, "training_log_critic.npy")
    np.save(log_path, {"train": np.array(train_hist), "val": np.array(val_hist)})
    
    print(f"  -> saved critic log: {log_path}")
    print(f"  -> saved final: {final_path}")


if __name__ == "__main__":
    main()