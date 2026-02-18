import os
import numpy as np
import copy

from sbto.solvers.solver_base import SolverState, SamplingBasedSolver
from sbto.data.utils import solver_state_path_from_rundir
from sbto.data.filenames import (
    INITIAL_SOLVER_STATE_SUFFIX,
    FINAL_SOLVER_STATE_SUFFIX,
    BEST_TRAJECTORY_FILENAME,
    )
def _get_state_from_rundir(rundir: str, solver: SamplingBasedSolver, suffix: str) -> SolverState:
    filename = solver_state_path_from_rundir(rundir, suffix)
    solver_state_file = os.path.join(rundir, filename)
    solver_state_0 = copy.deepcopy(solver.state)
    data = np.load(solver_state_file)
    for k, v in data.items():
        setattr(solver_state_0, k, v)
    return solver_state_0

def get_initial_state_from_rundir(rundir: str, solver: SamplingBasedSolver) -> SolverState:
    return _get_state_from_rundir(rundir, solver, INITIAL_SOLVER_STATE_SUFFIX)

def get_final_state_from_rundir(rundir: str, solver: SamplingBasedSolver) -> SolverState:
    return _get_state_from_rundir(rundir, solver, FINAL_SOLVER_STATE_SUFFIX)

def get_best_trajectory_from_rundir(rundir: str):
    path = os.path.join(rundir, BEST_TRAJECTORY_FILENAME)
    return np.load(path)