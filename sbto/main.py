import hydra

from utils.hydra import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sim, task, solver = instantiate_from_cfg(cfg)

    # Warm start
    if (
        cfg.warm_start.multiple_shooting or
        cfg.warm_start.incremental
        ):
        
        cfg_ws = set_cfg_warm_start(cfg)

        # Update solver Nit
        _N_it = cfg_ws.solver.cfg.N_it
        if cfg_ws.warm_start.N_it > 0:
            solver.cfg.N_it = cfg_ws.warm_start.N_it

        solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
        rundir = optimize_and_save_data(
            cfg_ws,
            sim,
            task,
            solver,
            hydra_rundir,
            solver_state_0=solver_state_0,
        )
        cfg.warm_start.rundir = rundir

        # Set back the number of solver iterations
        solver.cfg.N_it = _N_it
        # Reset warm start params
        cfg.warm_start.multiple_shooting = False
        cfg.warm_start.incremental = False

    # Optimize single shooting
    if solver.cfg.N_it > 0:
        solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
        rundir = optimize_and_save_data(
            cfg,
            sim,
            task,
            solver,
            hydra_rundir,
            solver_state_0,
        )
        
if __name__ == "__main__":
    main()