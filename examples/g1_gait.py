import os

<<<<<<< HEAD
from sbto.tasks.unitree_g1.g1_gait import G1_Gait
from sbto.mj.solver.cem_old import CEM
from sbto.mj.solver.efficient_cem import EfficientCEM
from sbto.utils.plotting import plot_state_control, plot_costs
from sbto.utils.viewer import render_and_save_trajectory
from sbto.utils.plot_contact import plot_contact_achieved_vs_planned


def main():
    T = 200
    Nknots = 10
    Nit = 200
    nlp = G1_Gait(T, Nknots, interp_kind="cubic", Nthread=-1)
    nlp._chunk_size = 2

    solver = EfficientCEM(
        nlp,
        N_samples=512,
        elite_frac=0.03,
        alpha_mean=0.95,
=======
from sbto.tasks.unitree_g1.g1_gait import G1_Gait, ConfigG1Gait
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments

def main():
    cfg_nlp = ConfigG1Gait(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=15
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.04,
        alpha_mean=0.9,
>>>>>>> origin/main
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=100,
        sigma0=0.2
    )
    run_experiments(
        G1_Gait,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="cem"
    )
<<<<<<< HEAD

    states, best_u, cost, all_costs = solver.solve(state, Nit)
    print("Best cost:", cost)



    result_dir = "./plots"
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
    plot_contact_achieved_vs_planned(obs_traj, nlp)

=======
>>>>>>> origin/main

if __name__ == "__main__":
    main()