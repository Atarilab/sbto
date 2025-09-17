import numpy as np

from sbto.tasks.unitree_g1.g1_gait import G1_Gait
from sbto.mj.solver.cem import CEM
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import visualize_trajectory

def main():
    T = 200
    Nknots = 10
    Nit = 500
    nlp = G1_Gait(T, Nknots, interp_kind="cubic")

    solver = CEM(
        nlp,
        N_samples=512,
        elite_frac=0.1,
        alpha_mean=0.9,
        alpha_cov=0.25,
        seed=42
        )
    state = solver.init_state(
        mean=None,
        cov=None,
        temperature=1.0,
        sigma_mult=1.0
    )

    states, best_u, cost, all_costs = solver.solve(state, Nit)
    print("Best cost:", cost)

    x_traj, u_traj, obs_traj, cost = solver.evaluate(best_u)
    plot_costs(all_costs)

    plot_state_control(
        x_traj[:, 0],
        x_traj[:, 1:],
        u_traj,
        best_u,
        nlp.Nq,
        nlp.Nu,
        )
    visualize_trajectory(nlp.mj_model, nlp.mj_data, x_traj[:, 0], x_traj[:, 1:])


if __name__ == "__main__":
    main()