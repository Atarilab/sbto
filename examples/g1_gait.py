import os

from sbto.tasks.unitree_g1.g1_gait import G1_Gait, ConfigG1Gait
from sbto.mj.solver.cem import CEM
from sbto.mj.solver.efficient_cem import EfficientCEM, EfficientCEMConfig
from sbto.utils.plotting import plot_state_control, plot_costs, plot_mean_cov
from sbto.utils.viewer import render_and_save_trajectory

def main():
    cfg_nlp = ConfigG1Gait(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=15
    )

    nlp = G1_Gait(cfg_nlp)
    nlp._chunk_size = 2

    cfg_solver = EfficientCEMConfig(
        N_samples=1024,
        elite_frac=0.03,
        alpha_mean=0.9,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=300,
    )

    solver = EfficientCEM(
        nlp,
        cfg_solver
        )
    state = solver.init_state(
        mean=None,
        cov=None,
        sigma_mult=0.4
    )

    states, best_u, cost, all_costs = solver.solve(state)
    print("Best cost:", cost)

    x_traj, u_traj, obs_traj, cost = solver.evaluate(best_u)
    result_dir = "./plots"
    cfg_solver.save_to_yaml(result_dir)
    cfg_nlp.save_to_yaml(result_dir)

    plot_mean_cov(
        x_traj[:, 0],
        states[-1].mean,
        best_u,
        states[-1].cov,
        u_traj,
        Nu=nlp.Nu,
        save_dir=result_dir,
    )

    render_and_save_trajectory(
        nlp.mj_model,
        nlp.mj_data,
        x_traj[:, 0],
        x_traj[:, 1:],
        save_path=result_dir
        )

    plot_costs(
        all_costs,
        save_dir=result_dir
        )

    plot_state_control(
        x_traj[:, 0],
        x_traj[:, 1:],
        u_traj,
        best_u,
        nlp.Nq,
        nlp.Nu,
        save_dir=result_dir
        )


if __name__ == "__main__":
    main()