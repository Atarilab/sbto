import hydra
from hydra.utils import instantiate
from typing import Optional
import copy
import os
import numpy as np

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.run.optimize import optimize_single_shooting, optimize_mutiple_shooting, optimize_cumulative_opt
from sbto.run.save import save_results, get_final_state_from_rundir

def optimize_and_save_data(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    solver_state_0: Optional[SolverState] = None,
    multiple_shooting: bool = False,
    cumul_opt: bool = False,
    ) -> None:

    # Save initial state
    if solver_state_0:
        solver_state_0 = copy.deepcopy(solver_state_0)
    else:
        solver_state_0 = copy.deepcopy(solver.state)

    # Single shooting or multiple_shooting
    if multiple_shooting:
        if not isinstance(task, TaskMjRef):
            raise ValueError("Task should be an instance of TaskMjRef (with reference)")
        optimizer_fn = optimize_mutiple_shooting
    elif cumul_opt:
        optimizer_fn = optimize_cumulative_opt
    else:
        optimizer_fn = optimize_single_shooting
    
    solver_state_final, all_samples, all_costs = optimizer_fn(
        sim,
        task,
        solver,
        solver_state_0
    )

    save_results(
        sim,
        task,
        solver_state_0,
        solver_state_final,
        all_samples,
        all_costs,
        description,
        hydra_rundir,
        save_fig,
        multiple_shooting,
    )

def instantiate_from_cfg(cfg):
    sim = instantiate(cfg.task.sim)
    task = instantiate(cfg.task, sim=sim)
    solver = instantiate(cfg.solver, D=sim.Nvars_u)
    return sim, task, solver

def get_initial_state_solver_from_ref(sim, task, solver):
    if not isinstance(task, TaskMjRef):
        print("Task has no reference.")
        return None
    qpos_from_ref = task.ref.act_qpos[sim.t_knots, :]
    pd_knots_from_ref = sim.scaling.inverse(qpos_from_ref).reshape(-1)
    solver_state_0 = solver.init_state(mean=pd_knots_from_ref)
    return solver_state_0

def get_warm_start_state_solver(cfg, sim, task, solver) -> SolverState:
    # Set initial solver state
    state_solver_0 = None
    if cfg.init_knots_from_ref and isinstance(task, TaskMjRef):
        state_solver_0 = get_initial_state_solver_from_ref(sim, task, solver)

    if cfg.warm_start.rundir and os.path.exists(cfg.warm_start.rundir):
        state_solver_0 = get_final_state_from_rundir(cfg.warm_start.rundir, solver)

        if cfg.warm_start.reset_sigma0:
            solver.init_state()
            state_solver_0.cov += solver.state.cov

        # Reset min cost/best
        state_solver_0.min_cost = np.inf
        state_solver_0.min_cost_all = np.inf
        D = len(state_solver_0.mean)
        state_solver_0.best = np.empty(D)
        state_solver_0.best_all = np.empty(D)

    return state_solver_0

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    sim, task, solver = instantiate_from_cfg(cfg)

    if cfg.warm_start.multiple_shooting:
        description = cfg.description + "warm_start_ms"
        rundir = optimize_and_save_data(
            sim,
            task,
            solver,
            description,
            hydra_rundir,
            cfg.save_fig,
            state_solver_0,
            multiple_shooting=True
        )
        cfg.warm_start.rundir = rundir

    if cfg.warm_start.cumul_opt:
        description = cfg.description + "warm_start_cumul_opt"
        rundir = optimize_and_save_data(
            sim,
            task,
            solver,
            description,
            hydra_rundir,
            cfg.save_fig,
            state_solver_0,
            cumul_opt=True
        )
        cfg.warm_start.rundir = rundir
    
    state_solver_0 = get_warm_start_state_solver(cfg, sim, task, solver)

    rundir = optimize_and_save_data(
        sim,
        task,
        solver,
        cfg.description,
        hydra_rundir,
        cfg.save_fig,
        state_solver_0,
    )
    
if __name__ == "__main__":
    main()