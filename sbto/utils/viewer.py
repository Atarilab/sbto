import time
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def visualize_trajectory(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    t: np.ndarray,
    x_traj: np.ndarray,
) -> None:
    """
    Visualizes the trajectory in a Mujoco viewer with pause and step-by-step control.
    - Space: toggle pause
    - Right arrow: step forward
    - Left arrow: step backward
    """
    T = len(x_traj)
    PAUSE_LOOP = 0.5
    dt_array = np.diff(t, append=0.)
    dt_array[-1] = PAUSE_LOOP  # pause at the end

    Nq = mj_model.nq
    step = 0
    paused = {"active": False}  # use dict so closure can mutate
    step_request = {"delta": 0}

    def key_callback(keycode: int):
        # Space bar
        if keycode == 32:  # space
            paused["active"] = not paused["active"]
        # Left arrow
        elif keycode == 263:
            step_request["delta"] = -1
            paused["active"] = True
        # Right arrow
        elif keycode == 262:
            step_request["delta"] = 1
            paused["active"] = True

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if paused["active"]:
                # Step manually only when requested
                if step_request["delta"] != 0:
                    step = (step + step_request["delta"]) % T
                    step_request["delta"] = 0
                else:
                    time.sleep(0.05)
                    continue
            else:
                # Play mode
                step = (step + 1) % T

            q, v = np.split(x_traj[step], [Nq])
            mj_data.qpos = q
            mj_data.qvel = v
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            time.sleep(dt_array[step])