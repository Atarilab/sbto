#cd /Users/mustaphadaly/Desktop/sbto
#mjpython examples/g1_gait_MPPI.py

import os

from sbto.tasks.unitree_g1.g1_gait import G1_Gait
from sbto.mj.solver.MPPI import MPPI
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import render_and_save_trajectory


def main():
    T = 200 # time horizon length
    Nknots = 10
    Nit = 200
    nlp = G1_Gait(T, Nknots, interp_kind="cubic", Nthread=-1)

    solver = MPPI(
    nlp,
    N_samples=512,
    lambda_=2,      # 0.5–2.0
    alpha_mean=0.9,   # 0.7–1.0; 
    alpha_cov=0.25,   # 0.1–0.3  
    seed=42
    )
    state = solver.init_state(
        mean=None,
        cov=None,
        temperature=0.5,
        sigma_mult=0.3 
    )

    print(f"[MPPI] lambda = {solver.lambda_}")
    states, best_u, cost, all_costs = solver.solve(state, Nit)
    print("Best cost:", cost)



    result_dir = "./plots_G1_MPPI"
    os.makedirs(result_dir, exist_ok=True)

    x_traj, u_traj, obs_traj, cost = solver.evaluate(best_u)

    render_and_save_trajectory(
    nlp.mj_model,
    nlp.mj_data,
    x_traj[:, 0],
    x_traj[:, 1:],
    save_path=os.path.join(result_dir, "g1_gait.mp4")  
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
