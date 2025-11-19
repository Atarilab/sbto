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

from sbto.utils.modulation import beta_mod, step_mod, mpc_mod, step_decayed_mod

def optimize_with_mod(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None,
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []
    pbar = trange(solver.cfg.N_it, desc="Optimizing w. cost modulation", leave=True)
    pbar_postfix = {}

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    N_it_full = 150
    mod_fn = step_mod

    start = time.time()
    for it in pbar:
        # decay = beta_mod(it, solver.cfg.N_it, sim.T, b_start=1.25)
        decay = mod_fn(it, solver.cfg.N_it, sim.T, Nknots=sim.Nknots)
        task.update_mod(decay)
        solver.update_mod(decay[sim.t_knots])

        samples = solver.get_samples()
        costs = compute_cost(samples, sim, task)
        solver.update(samples, costs)

        all_samples.append(samples)
        all_costs.append(costs)

        pbar_postfix["min_cost"] = solver.state.min_cost_all
        pbar_postfix["cost"] = solver.state.min_cost

        pbar.set_postfix(pbar_postfix)


    if N_it_full > 0:
        solver.state.min_cost_all=np.inf
        solver.state.best_all=np.empty_like(solver.state.mean)

        pbar = trange(N_it_full, desc="Optimizing full traj", leave=True)
        pbar_postfix = {}

        # Optimizing full cost
        decay = np.ones(sim.T)
        for it in pbar:
            task.update_mod(decay)
            solver.update_mod(decay[sim.t_knots])

            samples = solver.get_samples()
            costs = compute_cost(samples, sim, task)
            solver.update(samples, costs)

            all_samples.append(samples)
            all_costs.append(costs)

            pbar_postfix["min_cost"] = solver.state.min_cost_all
            pbar_postfix["cost"] = solver.state.min_cost
            pbar.set_postfix(pbar_postfix)

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, all_costs_arr