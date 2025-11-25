import time
import copy
import numpy as np
import numpy.typing as npt
from tqdm import trange
from typing import Tuple, Optional, Any

from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.utils.modulation import step_mod, step_mod_transition

Array = npt.NDArray[np.float64]

def compute_cost(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    return task.cost(*sim.rollout(u_knots)[1:])

def compute_cost_multiple_shooting(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    x_shooting = task.ref.x[sim.t_knots]
    return task.cost(*sim.rollout_multiple_shooting(u_knots, x_shooting)[1:])

def compute_cost_cumul(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    cumul_mod: Array,
    **kwargs,
    ):
    # s = time.time()
    task.update_mod(cumul_mod)
    solver.update_mod(cumul_mod[sim.t_knots])
    # e = time.time()
    # print("update_mod", e-s)

    id_zero = np.where(cumul_mod == 0)[0]
    if len(id_zero) == 0:
        return compute_cost(u_knots, sim, task)

    T_end = id_zero[0]-1
    # print("T_end", T_end)
    # print("t_knots", sim.t_knots)
    # print("cumul_mod, t_knots", cumul_mod[sim.t_knots])

    # s = time.time()
    # r = sim.rollout_t_steps(u_knots, T_end)
    # e = time.time()
    # print("rollout_t_steps", e-s)
    # s = time.time()
    # c = task.cost(*r[1:])
    # e = time.time()
    # print(c[:10])
    # print("cost", e-s)
    return task.cost(*sim.rollout_t_steps(u_knots, T_end)[1:])

def _optimize(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    compute_cost_fn: Any,
    cumul_opt: bool = False,
    init_state_solver: Optional[SolverState] = None,
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []
    pbar = trange(solver.cfg.N_it, desc="Optimizing", leave=True)
    pbar_postfix = {}

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    start = time.time()

    cumul_mod = np.ones(sim.T)
    try:
        task.update_mod(cumul_mod)
        solver.update_mod(cumul_mod[sim.t_knots])
    except:
        pass

    all_knots_optimized = True
    reset_best_knots_all = True

    for it in pbar:
        
        if cumul_opt:
            cumul_mod = step_mod(it, solver.cfg.N_it, sim.T, Nknots=sim.Nknots)
            all_knots_optimized = np.all(cumul_mod == 1.)

            # Reset best knots when all knots are optimized
            if reset_best_knots_all and all_knots_optimized:
                solver.state.min_cost_all = np.inf
                reset_best_knots_all = False

        samples = solver.get_samples()
        all_samples.append(samples.copy())

        costs = compute_cost_fn(samples, sim, task, solver=solver, cumul_mod=cumul_mod)
        all_costs.append(costs)

        solver.update(samples, costs)

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
        False,
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
        False,
        init_state_solver,
    )

# def optimize_cumulative_opt(
#     sim: SimRolloutBase,
#     task: OCPBase,
#     solver: SamplingBasedSolver,
#     init_state_solver: Optional[SolverState] = None 
#     ) -> Tuple[SolverState, Array, Array]:
#     return _optimize(
#         sim,
#         task,
#         solver,
#         compute_cost_cumul,
#         True,
#         init_state_solver,
#     )

def optimize_cumulative_opt(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None 
    ) -> Tuple[SolverState, Array, Array]:
    all_costs = []
    all_samples = []

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    start = time.time()

    cumul_mod = np.ones(sim.T)
    try:
        task.update_mod(cumul_mod)
        solver.update_mod(cumul_mod[sim.t_knots])
    except:
        pass

    all_knots_optimized = True
    reset_best_knots_all = True

    N_max_it_per_knots = 50
    min_std_next = 1.0e-2
    
    def get_max_diag_std(solver):
        return np.max(np.diag(solver.state.cov * solver.alpha_cov))
    
    Nmax_it = N_max_it_per_knots * sim.Nknots
    pbar = trange(Nmax_it, desc="Optimizing", leave=True)
    pbar_postfix = {}
    cumul_mod = np.zeros(sim.T)

    for N_knots_to_opt in range(1, sim.Nknots): # start with first 2 knots
        nit = 0

        t_end = sim.t_knots[N_knots_to_opt] + 1
        cumul_mod[:t_end] = 1.
        task.update_mod(cumul_mod)
        solver.update_mod(cumul_mod[sim.t_knots])
        all_knots_optimized = np.all(cumul_mod == 1.)

        while min_std_next < get_max_diag_std(solver) and nit < N_max_it_per_knots:
            nit += 1

            # Reset best knots when all knots are optimized
            if reset_best_knots_all and all_knots_optimized:
                solver.state.min_cost_all = np.inf
                reset_best_knots_all = False

            samples = solver.get_samples()
            all_samples.append(samples.copy())

            costs = compute_cost_cumul(samples, sim, task, solver=solver, cumul_mod=cumul_mod)
            all_costs.append(costs)

            solver.update(samples, costs)

            pbar.update(1)
            pbar_postfix["min_cost"] = solver.state.min_cost_all
            pbar_postfix["cost"] = solver.state.min_cost
            pbar_postfix["Nk_opt"] = N_knots_to_opt
            pbar_postfix["Nit"] = nit
            pbar.set_postfix(pbar_postfix)

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, all_costs_arr
