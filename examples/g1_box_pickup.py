import os

from sbto.tasks.unitree_g1.g1_box_pickup import G1_BoxPickup, ConfigG1BoxPickup
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments, sweep_param

def main():
    cfg_nlp = ConfigG1BoxPickup(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=8
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.05,
        alpha_mean=0.9,
        alpha_cov=0.15,
        seed=42,
        quasi_random=True,
        N_it=200,
        sigma0=0.25,
        b=1e-4,
        )
    sweep = sweep_param(
        cfg_solver,
        "seed",
        (0, 4),
        5   
    )
    run_experiments(
        G1_BoxPickup,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="sweep_seed"
    )

if __name__ == "__main__":
    main()