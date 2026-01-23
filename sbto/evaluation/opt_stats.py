def total_sim_timesteps(opt_stats_dict) -> int:
    n_knots = list(opt_stats_dict["iterations"].values())[-1]["n_knots_to_opt"]
    last_n_knots = False
    s = 0
    for it, stats in opt_stats_dict["iterations"].items():
        s += stats["n_sim_steps_rollout"]
        # if last_n_knots:
        #     break
        # if stats["n_knots_to_opt"] == n_knots:
        #     last_n_knots = True
    return s

def total_sim_timesteps_mpc(
    N_samples: int,
    rollout_per_steps: list[int],
    horizon: float,
    dt: float,
    ) -> int:
    """
    rollout_per_steps is the number of rollouts done per MPC optimization.
    """
    H = int(horizon / dt)
    return N_samples * sum(rollout_per_steps) * H