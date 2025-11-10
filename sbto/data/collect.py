import numpy as np
from typing import Optional
from copy import deepcopy, copy

from sbto.data.utils import save_all_samples_and_cost, save_all_states, create_dirs
from sbto.mj.nlp_mj import NLP_MuJoCo
from sbto.mj.solver_base import SamplingBasedSolver
from sbto.utils.config import ConfigBase
from sbto.data.randomize import RandomizedParamConfig, get_randomized_config

def collect_data(
    Nruns: int,
    nlp: NLP_MuJoCo,
    cfg_nlp: ConfigBase,
    solver: SamplingBasedSolver,
    cfg_solver: ConfigBase,
    cfg_param: Optional[RandomizedParamConfig] = None,
    init_state_solver: Optional[str] = None,
    randomize_initial_state: bool = True,
    randomize_solver_seed: bool = True,
    ):
    _description = ""

    if not cfg_param is None:
        all_cfg_nlp = get_randomized_config(cfg_nlp, cfg_param, num=Nruns)
    else:
        all_cfg_nlp = [copy(cfg_nlp) for _ in range(Nruns)]
    
    # run all configs
    for cfg_nlp in all_cfg_nlp:

            # create run dir
            exp_name = nlp.__name__
            rundir = create_dirs(exp_name, _description)

            # randomize seed
            if randomize_solver_seed:
                seed = np.random.randint(13*len(all_cfg_nlp))
                cfg_solver.seed = seed

            # save configs
            for c in [cfg_nlp, cfg_solver]:
                c.save(rundir)

            # run optimization
            n = nlp(cfg_nlp)
            s = solver(nlp=n, cfg=cfg_solver)
            if init_state_solver is None:
                state_solver = s.init_state()
            else:
                state_solver = init_state_solver

            # randomize initial state
            if randomize_initial_state:
                try:
                    n.randomize_initial_state()
                except:
                    print("Randomizing initial state failed.")
                    
            all_solver_states, best_qdes_knots, cost, all_costs, all_samples = s.solve(deepcopy(state_solver))

            save_all_samples_and_cost(rundir, all_samples, all_costs)