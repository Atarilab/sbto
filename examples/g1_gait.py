import os

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
        elite_frac=0.05,
        alpha_mean=0.9,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=150,
        sigma0=0.2,
        a=1e-5,
        b=1e-5,
    )
    run_experiments(
        G1_Gait,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="cem"
    )

if __name__ == "__main__":
    main()