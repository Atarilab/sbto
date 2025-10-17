import os

from sbto.tasks.unitree_g1.g1_gait import G1_Gait, ConfigG1Gait
from sbto.mj.solver.cem import CEM, CEMConfig, CEMBeta, CEMBetaConfig
from sbto.utils.exp_manager import run_experiments

def main():
    cfg_nlp = ConfigG1Gait(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=15
    )
    cfg_solver = CEMBetaConfig(
        N_samples=1024,
        elite_frac=0.04,
        alpha_mean=0.7,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=100,
        kappa=15.
    )
    run_experiments(
        G1_Gait,
        cfg_nlp,
        CEMBeta,
        cfg_solver,
        description="cem_b"
    )

if __name__ == "__main__":
    main()