import numpy as np
from math import ceil
from typing import List
from dataclasses import dataclass

@dataclass
class GaitConfig:
    n_feet: int
    stance_ratio: List  # shape: (n_feet,)
    phase_offset: List  # shape: (n_feet,)
    nominal_period: float     # seconds

def generate_contact_plan(T: int, dt: float, config: GaitConfig) -> np.ndarray:
    """
    Generate a binary contact plan for each foot over a horizon T.
    
    Args:
        T (int): Total number of time steps.
        dt (float): Time step duration.
        feet_names (List[str]): List of foot names.
        config (GaitConfig): Gait configuration.
        
    Returns:
        contact_plan (np.ndarray): Shape (T, n_feet), binary contact states.
    """
    nodes_per_cycle = round(config.nominal_period / dt)
    plan = np.zeros((T, config.n_feet), dtype=np.int8)

    for foot_id, (stance, offset) in enumerate(zip(config.stance_ratio, config.phase_offset)):
        make_contact_phase = offset
        break_contact_phase = (offset + stance) % 1.0

        for t in range(T):
            t_phase = (t % nodes_per_cycle) / nodes_per_cycle
            in_contact = False
            if make_contact_phase < break_contact_phase:
                in_contact = make_contact_phase <= t_phase < break_contact_phase
            else:
                in_contact = t_phase >= make_contact_phase or t_phase < break_contact_phase
            plan[t, foot_id] = int(in_contact)

    return plan

quad_jump = GaitConfig(
    4,
    stance_ratio=[0.75, 0.75, 0.75, 0.75],
    phase_offset=[0.0, 0.0, 0.0, 0.0],
    nominal_period=0.8
    )
quad_trot = GaitConfig(
    4,
    stance_ratio=[0.5, 0.5, 0.5, 0.5],
    phase_offset=[0.5, 0.0, 0.0, 0.5],
    nominal_period=0.5
    )
quad_bound = GaitConfig(
    4,
    stance_ratio=[0.5, 0.5, 0.5, 0.5],
    phase_offset=[0.5, 0.5, 0.0, 0.0],
    nominal_period=0.6
    )
humanoid_trot = GaitConfig(
    2,
    stance_ratio=[0.56, 0.56],
    phase_offset=[0.5, 0.0],
    nominal_period=0.9,
    )
humanoid_jump = GaitConfig(
    2,
    stance_ratio=[0.7, 0.7],
    phase_offset=[0.0, 0.0],
    nominal_period=0.6
    )