from sbto.tasks.unitree_go2.go2_pronk import Go2_Pronk, ConfigGo2Pronk
from sbto.mj.solver.cem import CEM, CEMConfig
from sbto.utils.exp_manager import run_experiments



def main():
    cfg_nlp = ConfigGo2Pronk(
        T=700,
        interp_kind="quadratic",
        Nthread=-1,
        Nknots=10, #best is 10 so far 
        scene="scene_position.xml",  
    )
    cfg_solver = CEMConfig(
        N_samples=1024,
        elite_frac=0.04,
        alpha_mean=0.9,
        alpha_cov=0.1,
        seed=42,
        quasi_random=True,
        N_it=100,
        sigma0=0.2,
    )

    run_experiments(
        Go2_Pronk,
        cfg_nlp,
        CEM,
        cfg_solver,
        description="cem_go2_pronk",
        
    )
    
if __name__ == "__main__":
    main()