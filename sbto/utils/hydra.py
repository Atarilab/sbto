from hydra.utils import instantiate
from typing import Optional
import copy
import os
import glob
import yaml
import numpy as np
from functools import partial
from omegaconf import OmegaConf

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.run.optimize import optimize_single_shooting, optimize_mutiple_shooting, optimize_incremental_opt
from sbto.run.save import save_results, get_final_state_from_rundir
from sbto.run.stats import OptimizationStats

def optimize_and_save_data(
    cfg,
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    hydra_rundir: str = "",
    solver_state_0: Optional[SolverState] = None,
    opt_stats: Optional[OptimizationStats] = None,
    ) -> str:

    # Copy initial state
    if solver_state_0:
        solver_state_0 = copy.deepcopy(solver_state_0)
    else:
        solver_state_0 = copy.deepcopy(solver.state)

    # Randomize time of the knots for data diversity
    if cfg.warm_start.rand_t_knots > 0:
        sim.randomize_t_knots(cfg.warm_start.rand_t_knots, solver.cfg.seed)

    # Multiple_shooting
    if cfg.warm_start.multiple_shooting:
        if not isinstance(task, TaskMjRef):
            raise ValueError("Task should be an instance of TaskMjRef (with reference)")
        optimizer_fn = optimize_mutiple_shooting

    # Incremental opt
    elif cfg.warm_start.incremental:
        optimizer_fn = partial(
            optimize_incremental_opt,
            N_max_it_per_knots=cfg.warm_start.N_max_incr,
            min_std_next=cfg.warm_start.min_std_next,
            min_std_final=cfg.warm_start.min_std_final,
        )

    # Single shooting
    else:
        optimizer_fn = optimize_single_shooting
    
    solver_state_final, all_samples, all_costs, opt_stats = optimizer_fn(
        sim,
        task,
        solver,
        solver_state_0,
        opt_stats,
    )

    rundir = save_results(
        sim,
        task,
        solver_state_0,
        solver_state_final,
        all_samples,
        all_costs,
        cfg.exp_name,
        cfg.description,
        hydra_rundir,
        cfg.data_processing.save_fig,
        cfg.data_processing.save_video,
        cfg.data_processing.save_samples_costs,
        cfg.warm_start.multiple_shooting,
        cfg.data_processing.split_state,
        cfg.data_processing.save_top,
        cfg.data_processing.remove_keys,
    )

    opt_stats.save(rundir)

    return rundir, opt_stats

def instantiate_from_cfg(cfg):
    sim = instantiate(cfg.task.sim)
    task = instantiate(cfg.task, sim=sim)
    random = instantiate(cfg.random, sim=sim, seed=cfg.solver.cfg.seed)
    solver = instantiate(cfg.solver, D=sim.Nvars_u)
    return sim, task, solver, random

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
    solver_state_0 = None
    if cfg.init_knots_from_ref and isinstance(task, TaskMjRef):
        solver_state_0 = get_initial_state_solver_from_ref(sim, task, solver)

    if cfg.warm_start.rundir and os.path.exists(cfg.warm_start.rundir):
        solver_state_0 = get_final_state_from_rundir(cfg.warm_start.rundir, solver)

        if not cfg.warm_start.cp_best:
            solver.reset_min_cost_best(solver_state_0)

        if cfg.warm_start.add_cov_diag > 0.:
            solver.init_state()
            N = solver_state_0.mean.shape[0]
            solver_state_0.cov += cfg.warm_start.add_cov_diag * np.eye(N)

    return solver_state_0

def set_cfg_warm_start(cfg):
    cfg_ws = copy.deepcopy(cfg)
    WARM_START_MULTIPLE_SHOOTING = "ws_ms"
    WARM_START_INCREMENTAL = "ws_incr"

    # Update description
    sep = "_" if cfg.description else ""
    if cfg_ws.warm_start.incremental:
        cfg_ws.description += sep + WARM_START_INCREMENTAL
    
    elif cfg_ws.warm_start.multiple_shooting:
        cfg_ws.description += sep + WARM_START_MULTIPLE_SHOOTING
    return cfg_ws

def get_optimization_stats_warm_start(cfg) -> OptimizationStats | None:
    rundir = cfg.warm_start.rundir
    if rundir and os.path.exists(rundir):
        opt_stats = OptimizationStats.load(rundir)
    else:
        opt_stats = None
    return opt_stats

def load_yaml(yaml_path):
    d = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
    return d

def save_yaml(yaml_path, data):
    if os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

def update_cfg_from_warm_start(cfg, hydra_rundir: str):
    rundir = cfg.warm_start.rundir
    
    if rundir and os.path.exists(rundir):
        
        # update config params from warm_start config
        cfg_paths = glob.glob(
            f"{rundir}/**/config.yaml",
            include_hidden=True,
            recursive=True
        )
        if len(cfg_paths) > 0:
            cfg_dict = load_yaml(cfg_paths[0])
            # copy motion path and Nknots
            cfg_warm_start = OmegaConf.create(cfg_dict)
            cfg.task.cfg_ref.motion_path = cfg_warm_start.task.cfg_ref.motion_path
            # cfg.task.sim.cfg.Nknots = cfg_warm_start.task.sim.cfg.Nknots
            # save yaml

            current_cfg_path = glob.glob(
                f"{hydra_rundir}/**/config.yaml",
                include_hidden=True,
                recursive=True
            )[0]
            save_yaml(current_cfg_path, OmegaConf.to_object(cfg))
