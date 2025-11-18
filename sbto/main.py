import hydra
from hydra.utils import instantiate
from typing import Optional

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.run.optimize import optimize_single_shooting
from sbto.run.save import save_results

def optimize_and_save_data(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    init_state_solver: Optional[SolverState] = None 
    ) -> None:

    solver_state, all_samples, all_costs = optimize_single_shooting(
        sim,
        task,
        solver,
        init_state_solver
    )
    save_results(
        sim,
        task,
        solver_state,
        all_samples,
        all_costs,
        description,
        hydra_rundir,
        save_fig,
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    sim, task, solver = instantiate_from_cfg(cfg)
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    init_state_solver = None
    if cfg.init_knots_from_ref and isinstance(task, TaskMjRef):
        init_state_solver = get_initial_state_solver_from_ref(sim, task, solver)

    optimize_and_save_data(
        sim,
        task,
        solver,
        cfg.description,
        hydra_rundir,
        cfg.save_fig,
        init_state_solver
    )
    
if __name__ == "__main__":
    main()