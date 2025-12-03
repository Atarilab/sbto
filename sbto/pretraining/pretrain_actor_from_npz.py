import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from rsl_rl.networks import MLP, EmpiricalNormalization
from mjlab.third_party.isaaclab.isaaclab.utils.math import matrix_from_quat

def quat_to_6d(q_np: np.ndarray) -> np.ndarray:
    """
    q_np: (..., 4) quaternion (w, x, y, z)
    returns: (..., 6) = first two columns of 3x3 rotation matrix, flattened
    """
    # to torch, float32
    q = torch.from_numpy(q_np.astype(np.float32))

    # normalization
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)

	# Convert quat to 3×3 rotation matrix
    R = matrix_from_quat(q)

    # 6-D representation
    R_2 = R[..., :, :2].reshape(*q.shape[:-1], 6)

    return R_2.numpy()
# Dataset

class SbtoNpzDataset(Dataset):
    """
    Build MJLAB policy inputs from your pretraining NPZ:
    joint_pos, joint_vel, error_anchor_b, base_ang_vel, joint_pos_rel,
    actions, object_pos_b, object_position_error, object_orientation_error.
    """
    def __init__(self, npz_path: str):
        super().__init__()
        data = np.load(npz_path)

        # ---- load from your current npz ----
        joint_pos  = data["joint_pos"].astype(np.float32)              # (N, T, 29)
        joint_vel  = data["joint_vel"].astype(np.float32)              # (N, T, 29)
        err_anchor_b = data["error_anchor_b"].astype(np.float32)       # (N, T, 6)
        base_ang_vel = data["base_ang_vel"].astype(np.float32)         # (N, T, 3)
        joint_pos_rel = data["joint_pos_rel"].astype(np.float32)       # (N, T, 29)
        actions    = data["actions"].astype(np.float32)                # (N, T, 29)
        obj_pos_b  = data["object_pos_b"].astype(np.float32)           # (N, T, 3)
        obj_pos_err = data["object_position_error"].astype(np.float32) # (N, T, 3)
        obj_ori_err = data["object_orientation_error"].astype(np.float32) # (N, T, 6)

        # !!! you MUST have some target in this npz, e.g. "u_policy"
        # If it's not there, you need to add it when you create the npz.
        raw_u = data["u_policy"].astype(np.float32)   # (N, T, U)
        N, T, A = joint_pos.shape
        U = raw_u.shape[2]
        self.U = U

        if T < 2:
            raise ValueError("Need at least 2 timesteps to build t->t+1 pairs")

        self.N_traj = N
        self.T_traj = T
        self.A = A

        # effective time steps (use t = 0..T-2)
        T_eff = T - 1

  
        cmd_t = np.concatenate(
            [joint_pos[:, :-1, :], joint_vel[:, :-1, :]],
            axis=-1
        )  # (N, T-1, 58)

        # 1) motion_anchor_ori_b
        motion_anchor_ori_b_t = err_anchor_b[:, :-1, :]       # (N, T-1, 6)

        # 2) base_ang_vel
        base_ang_vel_t = base_ang_vel[:, :-1, :]              # (N, T-1, 3)

        # 3) joint_pos term -> relative joint pos
        joint_pos_term_t = joint_pos_rel[:, :-1, :]           # (N, T-1, 29)

        # 4) joint_vel term -> actual joint_vel
        joint_vel_term_t = joint_vel[:, :-1, :]               # (N, T-1, 29)

        # 5) actions
        actions_t = actions[:, :-1, :]                        # (N, T-1, 29)

        # 6) object_global_pos
        obj_global_pos_t = obj_pos_b[:, :-1, :]               # (N, T-1, 3)

        # 7) object_pos_error
        obj_pos_err_t = obj_pos_err[:, :-1, :]                # (N, T-1, 3)

        # 8) object_ori_error
        obj_ori_err_t = obj_ori_err[:, :-1, :]                # (N, T-1, 6)

        # ---- flatten to samples ----
        self.command             = cmd_t.reshape(N * T_eff, 58)
        self.motion_anchor_ori_b = motion_anchor_ori_b_t.reshape(N * T_eff, 6)
        self.base_ang_vel        = base_ang_vel_t.reshape(N * T_eff, 3)
        self.joint_pos_term      = joint_pos_term_t.reshape(N * T_eff, A)
        self.joint_vel_term      = joint_vel_term_t.reshape(N * T_eff, A)
        self.actions_term        = actions_t.reshape(N * T_eff, A)
        self.object_global_pos   = obj_global_pos_t.reshape(N * T_eff, 3)
        self.object_pos_error    = obj_pos_err_t.reshape(N * T_eff, 3)
        self.object_ori_error    = obj_ori_err_t.reshape(N * T_eff, 6)

        # targets at time t
        target_u = raw_u[:, :-1, :]                            # (N, T-1, U)
        self.targets = target_u.reshape(N * T_eff, U)          # (N*(T-1), U)

        # obs dimension (should be 166)
        self.obs_dim = (
            self.command.shape[1]
            + self.motion_anchor_ori_b.shape[1]
            + self.base_ang_vel.shape[1]
            + self.joint_pos_term.shape[1]
            + self.joint_vel_term.shape[1]
            + self.actions_term.shape[1]
            + self.object_global_pos.shape[1]
            + self.object_pos_error.shape[1]
            + self.object_ori_error.shape[1]
        )

        self.num_samples = N * T_eff

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = {
            "command":             torch.from_numpy(self.command[idx]),             # (58,)
            "motion_anchor_ori_b": torch.from_numpy(self.motion_anchor_ori_b[idx]),# (6,)
            "base_ang_vel":        torch.from_numpy(self.base_ang_vel[idx]),       # (3,)
            "joint_pos":           torch.from_numpy(self.joint_pos_term[idx]),     # (29,)
            "joint_vel":           torch.from_numpy(self.joint_vel_term[idx]),     # (29,)
            "actions":             torch.from_numpy(self.actions_term[idx]),       # (29,)
            "object_global_pos":   torch.from_numpy(self.object_global_pos[idx]),  # (3,)
            "object_pos_error":    torch.from_numpy(self.object_pos_error[idx]),   # (3,)
            "object_ori_error":    torch.from_numpy(self.object_ori_error[idx]),   # (6,)
        }
        target = torch.from_numpy(self.targets[idx])                               # (U,)
        return obs, target



def collate_to_actor_obs(batch):
    """
    Concatenate in EXACT MJLAB order:
    command,
    motion_anchor_ori_b,
    base_ang_vel,
    joint_pos,
    joint_vel,
    actions,
    object_global_pos,
    object_pos_error,
    object_ori_error
    """
    cmd   = torch.stack([b[0]["command"]             for b in batch], dim=0)
    ori_b = torch.stack([b[0]["motion_anchor_ori_b"] for b in batch], dim=0)
    bang  = torch.stack([b[0]["base_ang_vel"]        for b in batch], dim=0)
    jpos  = torch.stack([b[0]["joint_pos"]           for b in batch], dim=0)
    jvel  = torch.stack([b[0]["joint_vel"]           for b in batch], dim=0)
    acts  = torch.stack([b[0]["actions"]             for b in batch], dim=0)
    obj_g = torch.stack([b[0]["object_global_pos"]   for b in batch], dim=0)
    obj_pe = torch.stack([b[0]["object_pos_error"]   for b in batch], dim=0)
    obj_oe = torch.stack([b[0]["object_ori_error"]   for b in batch], dim=0)

    actor_obs = torch.cat(
        [cmd, ori_b, bang, jpos, jvel, acts, obj_g, obj_pe, obj_oe],
        dim=-1,
    )  # (B, 166)

    targets   = torch.stack([b[1] for b in batch], dim=0)
    return actor_obs, targets



class ActorMLP(nn.Module):
    """
    Matches MJLAB actor head: 512, 256, 128 ELU MLP 
    Uses EmpiricalNormalization on inputs like in MJLAB 
    """
    def __init__(self, obs_dim: int, num_actions: int, hidden=(512, 256, 128), activation="elu"):
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
        model.norm.update(actor_obs)
        pred = model(actor_obs)  # (B, A)
        loss = nn.functional.mse_loss(pred, targets) 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        total += loss.item() * actor_obs.size(0) 
        n += actor_obs.size(0)
    return total / max(1, n) #average loss 


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
    ap.add_argument("--npz", type=str, required=True)
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

    obs_dim = ds.obs_dim           
    num_actions = ds.targets.shape[1]
    
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
    
    model = ActorMLP(
    obs_dim=obs_dim,
    num_actions=num_actions,
    hidden=(512, 256, 128),
    activation="elu",
    ).to(device)
    
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
                "num_actions": num_actions,
                "hidden":  (512, 256, 128),
                "activation": "elu",
                "val_mse": best_val,
            }, save_path)
            print(f"  -> saved: {save_path}")

    # also save final
    final_path = os.path.join(args.save_dir, "actor_from_sbto_last.pth")
    torch.save({
        "actor_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "hidden":  (512, 256, 128),
        "activation": "elu",
        "val_mse": best_val,
    }, final_path)
    
    #  network architecture summary 
    arch_path = os.path.join(args.save_dir, "network_architecture.txt")
    with open(arch_path, "w") as f:
        f.write(str(model))
    print(f"Saved model architecture to {arch_path}")

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