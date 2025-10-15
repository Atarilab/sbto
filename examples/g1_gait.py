import os

from sbto.tasks.unitree_g1.g1_gait import G1_Gait, ConfigG1Gait
from sbto.mj.solver.cem import CEM
from sbto.mj.solver.efficient_cem import EfficientCEM, EfficientCEMConfig
from sbto.utils.plotting import plot_state_control, plot_costs, plot_mean_cov, plot_contact_plan
from sbto.utils.viewer import render_and_save_trajectory
from sbto.utils.exp_manager import run_experiments

def main():
    cfg_nlp = ConfigG1Gait(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=15
    )
    nlp = G1_Gait(cfg_nlp)
    cfg_solver = EfficientCEMConfig(
        N_samples=1024,
        elite_frac=0.03,
        alpha_mean=0.9,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=200,
    )
    solver = EfficientCEM(
        nlp,
        cfg_solver
        )
    init_state = solver.init_state(
        mean=None,
        cov=None,
        sigma_mult=0.4
    )

    run_experiments(
        G1_Gait,
        cfg_nlp,
        EfficientCEM,
        cfg_solver,
        init_state,
        description="test"
    )

if __name__ == "__main__":
    main()