import time
import copy
import numpy as np
import numpy.typing as npt
from tqdm import trange
from typing import Tuple, Optional, Any

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState

Array = npt.NDArray[np.float64]

def compute_cost(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    ):
    return task.cost(*sim.rollout(u_knots)[1:])

def compute_cost_multiple_shooting(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    ):
    x_shooting = task.ref.x[sim.t_knots]
    return task.cost(*sim.rollout_multiple_shooting(u_knots, x_shooting)[1:])

def _optimize(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    compute_cost_fn: Any,
    init_state_solver: Optional[SolverState] = None,
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []
    pbar = trange(solver.cfg.N_it, desc="Optimizing", leave=True)
    pbar_postfix = {}

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    start = time.time()
    for _ in pbar:
        samples = solver.get_samples()
        costs = compute_cost_fn(samples, sim, task)
        solver.update(samples, costs)

        all_samples.append(samples)
        all_costs.append(costs)

        pbar_postfix["min_cost"] = solver.state.min_cost_all
        pbar.set_postfix(pbar_postfix)

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, all_costs_arr

def optimize_single_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None 
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost,
        init_state_solver, 
    )

def optimize_mutiple_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None 
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost_multiple_shooting,
        init_state_solver,
    )