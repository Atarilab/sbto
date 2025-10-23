import os
from sbto.tasks.unitree_go2.go2_gait import Go2_Gait  
from sbto.mj.solver.cem_old import CEM
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import render_and_save_trajectory
import mujoco


def main():
    T = 400
    Nknots = 10     # more interpolation points for gait
    Nit = 200       # more optimization iterations
    nlp = Go2_Gait(T, Nknots, interp_kind="cubic", Nthread=-1)
    nlp._chunk_size = 2

    solver = CEM(
        nlp,
        N_samples=512,
        elite_frac=0.03,
        alpha_mean=0.95,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
    )

    state = solver.init_state(mean=None, cov=None, sigma_mult=0.3)
    states, best_u, cost, all_costs = solver.solve(state, Nit)
    print("Best cost:", cost)

    result_dir = "./plots_go2_gait"
    os.makedirs(result_dir, exist_ok=True)

    x_traj, u_traj, obs_traj, cost = solver.evaluate(best_u)

    render_and_save_trajectory(
        nlp.mj_model,
        nlp.mj_data,
        x_traj[:, 0],
        x_traj[:, 1:],
        save_path=os.path.join(result_dir, "Go2_Gait.mp4")  
    )

    plot_costs(all_costs, save_dir=result_dir)
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