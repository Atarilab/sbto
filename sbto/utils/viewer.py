import time
import mujoco
import mujoco.viewer
import numpy as np
import numpy.typing as npt
import mujoco
import numpy as np
import cv2
import os

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
    t = np.squeeze(t)
    x_traj = np.squeeze(x_traj)
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

def render_and_save_trajectory(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    t: np.ndarray,
    x_traj: np.ndarray,
    save_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
):
    """
    Render a trajectory in MuJoCo using mujoco.Renderer and save it as a video.

    Args:
        mj_model: MuJoCo model.
        mj_data: MuJoCo data (will be modified during rendering).
        t: (T,) time array.
        x_traj: (T, n_state) array of states [qpos, qvel].
        save_video_path: Path to save the video (e.g., 'videos/output.mp4').
        fps: Frames per second for video.
        width: Render width.
        height: Render height.
        camera: Optional camera name (string) from the MJCF model.
    """
    # Initialize renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    dt = t[1] - t[0]
    fps = min(fps, int(1/dt))

    # Check save path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.isdir(save_path):
        filename = "trajectory_vis.mp4"
        save_path = os.path.join(save_path, filename)

    # Prepare video writer
    writer = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height),
    )

    nq = mj_model.nq
    nv = mj_model.nv
    n_frames = 0
    
    for i, timestep in enumerate(np.squeeze(t)):
        # Set MuJoCo state
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[:] = x_traj[i, :nq]
        mj_data.qvel[:] = x_traj[i, nq:nq + nv]
        mujoco.mj_forward(mj_model, mj_data)

        # Render frame
        if n_frames < (timestep * fps):
            n_frames += 1
            renderer.update_scene(mj_data, camera="track")
            frame = renderer.render()
            # Convert RGB to BGR for OpenCV and write
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)

    writer.release()
    renderer.close()
    print(f"Saved video to {save_path}")
