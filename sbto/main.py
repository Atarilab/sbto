import hydra

from sbto.utils.hydra import *

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sim, task, solver = instantiate_from_cfg(cfg)

    is_warm_start = (
        cfg.warm_start.multiple_shooting or
        cfg.warm_start.incremental
        )
    is_warm_start_only = is_warm_start and solver.cfg.N_it == 0

    # Warm start
    if is_warm_start:
        
        cfg_ws = set_cfg_warm_start(cfg)

        # Figures will be saved after warm_start
        if not is_warm_start_only:
            cfg_ws.save_fig = False
            cfg_ws.save_video = False
            cfg_ws.save_samples_costs = False

        # Update solver Nit
        _N_it = cfg_ws.solver.cfg.N_it
        if cfg_ws.warm_start.N_it > 0:
            solver.cfg.N_it = cfg_ws.warm_start.N_it

        solver_state_0 = get_warm_start_state_solver(cfg, sim, task, solver)
        opt_stats = get_optimization_stats_warm_start(cfg)

        rundir, opt_stats = optimize_and_save_data(
            cfg_ws,
            sim,
            task,
            solver,
            hydra_rundir,
            solver_state_0,
            opt_stats,
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
        opt_stats = get_optimization_stats_warm_start(cfg)

        rundir = optimize_and_save_data(
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