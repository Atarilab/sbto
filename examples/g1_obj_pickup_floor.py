import os

from sbto.tasks.unitree_g1.g1_obj_pickup_floor import G1_ObjPickupFloor, ConfigG1ObjPickupFloor
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments, sweep_param

def main():
    cfg_nlp = ConfigG1ObjPickupFloor(
        T=200,
        interp_kind="quadratic",
        Nthread=112,
        Nknots=14
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.05,
        alpha_mean=0.9,
        alpha_cov=0.2,
        seed=42,
        quasi_random=True,
        N_it=100,
        sigma0=0.3
        )
    sweep_seed = sweep_param(
        cfg_solver,
        "seed",
        range=(0, 9),
        num=10
    )
    run_experiments(
        G1_ObjPickupFloor,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="test"
    )

if __name__ == "__main__":
    main()