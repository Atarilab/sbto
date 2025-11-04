import os
import numpy as np
from typing import List

from sbto.data.utils import ALL_SAMPLES_COSTS_FILENAME, save_rollout

def get_all_costs_and_samples_paths(data_dir: str) -> List[str]:
    all_costs_samples_paths = []
    for exp_dir in os.listdir(data_dir):
        exp_path = os.path.join(data_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        costs_samples_path = os.path.join(
            exp_path,
            f"{ALL_SAMPLES_COSTS_FILENAME}.npz"
        )
        all_costs_samples_paths.append(costs_samples_path)
    
    return all_costs_samples_paths

def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    top_quantile: float
    ) -> np.ndarray:
    assert 0 < top_quantile <= 1.0, "top_quantile must be in (0, 1]."
    assert costs.ndim == 2, "Expected costs of shape (N, T)."
    assert samples.ndim == 3, "Expected samples of shape (N, T, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)

    N_total = costs_flat.shape[0]
    N_top = max(1, int(np.ceil(N_total * top_quantile)))

    # Use partial sort for efficiency
    idx_top = np.argpartition(costs_flat, N_top - 1)[:N_top]

    # Select top samples and sort by cost
    top_costs = costs_flat[idx_top]
    top_samples = samples_flat[idx_top]
    sorted_idx = np.argsort(top_costs)

    return top_samples[sorted_idx], top_costs[sorted_idx]

def aggregate_top_samples(
    nlp,
    data_dir: str,
    top_quantile: float = 0.01,
    ):
    all_costs_samples_paths = get_all_costs_and_samples_paths(data_dir)
    all_costs = []
    all_samples = []

    for path in all_costs_samples_paths:
        file = np.load(path)
        all_costs.append(file["costs"])
        all_samples.append(file["samples"])

    all_samples_arr = np.vstack(all_samples)
    all_costs_arr = np.vstack(all_costs)

    top_samples, top_costs = get_top_samples(all_costs_arr, all_samples_arr, top_quantile)    
    state_traj, u_traj, obs_traj = nlp.rollout_get_traj_with_x0(top_samples)

    # save rollout data
    save_rollout(
        data_dir,
        time=state_traj[:, :,  :1],
        x_traj=state_traj[:, :, 1:],
        u_traj=u_traj,
        obs_traj=obs_traj,
        costs=top_costs
    )