import os
import glob
import numpy as np
from typing import List

from sbto.data.filenames import ALL_SAMPLES_COSTS_FILENAME, ROLLOUT_FILENAME

def get_all_costs_and_samples_paths(data_dir: str) -> List[str]:
    all_costs_samples_paths = []
    all_costs_samples_paths += glob.glob(
        f"{data_dir}/*/{ALL_SAMPLES_COSTS_FILENAME}.*",
        include_hidden=True,
        recursive=True
        )
    all_costs_samples_paths += glob.glob(
        f"{data_dir}/{ALL_SAMPLES_COSTS_FILENAME}.*",
        include_hidden=True,
        recursive=True
        )
    return all_costs_samples_paths

def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    N_top_samples: float
    ) -> np.ndarray:
    assert costs.ndim == 2, "Expected costs of shape (N, T)."
    assert samples.ndim == 3, "Expected samples of shape (N, T, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)
    
    # Remove samples in double (in case keep elites for instance)
    costs_flat_unique, arg_unique = np.unique(costs_flat, return_index=True, sorted=True)
    samples_flat_unique = samples_flat[arg_unique]

    return samples_flat_unique[:N_top_samples], costs_flat_unique[:N_top_samples]

def aggregate_top_samples(
    sim,
    data_dir: str,
    N_top_samples: int = 0,
    percentile_cost: float = 0.,
    min_iteration: int = 0,
    save_data: bool = True
    ) -> str:
    all_costs_samples_paths = get_all_costs_and_samples_paths(data_dir)
    all_costs = []
    all_samples = []

    first_shape = 0
    for path in all_costs_samples_paths:
        file = np.load(path)
        samples, costs = file["samples"], file["costs"]
        if first_shape == 0:
            first_shape = samples.shape[-1]
        # Could be that same ref has runs with different number of knots
        if samples.shape[-1] != first_shape:
            continue
        all_costs.append(costs)
        all_samples.append(samples)

    if len(all_costs) > 1:
        all_samples_arr = np.vstack(all_samples)[min_iteration:, :]
        all_costs_arr = np.vstack(all_costs)[min_iteration:, :]
    elif len(all_costs) == 1:
        all_samples_arr = all_samples[0][min_iteration:, :]
        all_costs_arr = all_costs[0][min_iteration:, :]
    else:
        print("No data found.")
        return {}

    if percentile_cost > 0. and N_top_samples == 0:
        costs_flat = costs.reshape(-1)
        threshold = np.percentile(costs_flat, percentile_cost)
        top_mask = costs_flat <= threshold
        N_top_samples = np.sum(top_mask)
        print(f"Percentile cost: {percentile_cost} -> N_top_samples: {N_top_samples}")
    else:
        raise ValueError("Cost percentile or number of top samples need to be set.")

    top_samples, top_costs = get_top_samples(all_costs_arr, all_samples_arr, N_top_samples)
    # Rollout to get data
    try:
        t, state_traj, u_traj, obs_traj = sim.rollout(top_samples, with_x0=True)
    except:
        print(f"Rollout failed ({data_dir})")
        return {}
        ## Sanity check (should have the same cost as during the optimization)
        # c = task.cost(state_traj, u_traj, obs_traj)
    print(f"Min/Max costs: {top_costs.min():.2f} / {top_costs.max():.2f}")
    # print(c.min(), c.max())

    data = {
        "time": t,
        "x": state_traj,
        "u": u_traj,
        "o": obs_traj,
        "c": top_costs,
    }

    if save_data:
        rollout_data_path = os.path.join(data_dir, f"{ROLLOUT_FILENAME}.npz")
        np.savez(
            rollout_data_path,
            **data
        )

    return data
