from sbto.tasks.unitree_g1.g1_stand import G1_Stand
from sbto.mj.solver.cem import CEM
import numpy as np

def main():
    T = 200
    Nknots = 20
    Nit = 50
    nlp = G1_Stand(T, Nknots)

    solver = CEM(nlp, N_samples=1024, elite_frac=0.1, alpha_mean=0.8, alpha_cov=0.02, seed=42)
    state = solver.init_state(
        mean=None,
        cov=None,
        temperature=1.0,
        sigma_mult=1.0
    )

    states, costs, best_u = solver.solve(state, Nit)

    print("Best cost:", costs)

if __name__ == "__main__":
    main()