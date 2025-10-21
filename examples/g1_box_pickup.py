import os

from sbto.tasks.unitree_g1.g1_box_pickup import G1_BoxPickup, ConfigG1BoxPickup
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments

def main():
    cfg_nlp = ConfigG1BoxPickup(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=8
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.04,
        alpha_mean=0.9,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=10,
        sigma0=0.2
        )
    run_experiments(
        G1_BoxPickup,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="test"
    )

if __name__ == "__main__":
    main()