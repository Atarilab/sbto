import hydra

from sbto.utils.hydra import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    update_cfg_from_warm_start(cfg, hydra_rundir)
    sim, task, solver = instantiate_from_cfg(cfg)

    solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
    opt_stats = get_optimization_stats_warm_start(cfg)

    optimize_and_save_data(
        cfg,
        sim,
        task,
        solver,
        hydra_rundir,
        solver_state_0,
        opt_stats,
    )
        
if __name__ == "__main__":
    main()