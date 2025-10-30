import os

from sbto.tasks.unitree_g1.g1_obj_pickup_table import G1_ObjPickupTable, ConfigG1ObjPickupTable
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments, sweep_param

def main():
    cfg_nlp = ConfigG1ObjPickupTable(
        T=200,
        interp_kind="linear",
        Nthread=112,
        Nknots=6
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.05,
        alpha_mean=0.95,
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
        G1_ObjPickupTable,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="sweep_seed"
    )

if __name__ == "__main__":
    main()