import numpy as np

from sbto.tasks.unitree_g1.g1_stand import G1_Stand
from sbto.mj.solver.MPPI import MPPI
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import visualize_trajectory
from sbto.utils.viewer import render_and_save_trajectory

def main():
    T = 200
    Nknots = 5
    Nit = 25
    nlp = G1_Stand(T, Nknots, interp_kind="cubic")

    solver = MPPI(
    nlp,
    N_samples=512,
    lambda_=1.4,     # 0.5–2.0: Higher λ = broader weighting (more exploration); lower λ = greedier updates.
    alpha_mean=0.9,   # 0.7–1.0
    alpha_cov=0.25,   # 0.1–0.3
    seed=42
    )
    print(f"Lambda value: {solver.lambda_}")
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
    #visualize_trajectory(nlp.mj_model, nlp.mj_data, x_traj[:, 0], x_traj[:, 1:])


    render_and_save_trajectory(
    nlp.mj_model,
    nlp.mj_data,
    x_traj[:, 0],
    x_traj[:, 1:],
    filename="g1_gait.mp4"
)
    
if __name__ == "__main__":
    main()