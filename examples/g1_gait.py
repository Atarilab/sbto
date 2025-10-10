import os

from sbto.tasks.unitree_g1.g1_gait import G1_Gait
from sbto.mj.solver.cem import CEM
from sbto.mj.solver.efficient_cem import EfficientCEM
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import render_and_save_trajectory

def main():
    T = 200
    Nknots = 10
    Nit = 200
    nlp = G1_Gait(T, Nknots, interp_kind="cubic", Nthread=100)
    nlp._chunk_size = 2

    solver = EfficientCEM(
        nlp,
        N_samples=1024,
        elite_frac=0.03,
        alpha_mean=0.95,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        )
    state = solver.init_state(
        mean=None,
        cov=None,
        sigma_mult=0.3
    )

    states, best_u, cost, all_costs = solver.solve(state, Nit)
    print("Best cost:", cost)

    result_dir = "./plots"
    x_traj, u_traj, obs_traj, cost = solver.evaluate(best_u)

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