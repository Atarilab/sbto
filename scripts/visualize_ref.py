import hydra
import mujoco

from sbto.main import instantiate_from_cfg
from sbto.utils.viewer import visualize_trajectory

@hydra.main(version_base=None, config_path="../sbto/conf", config_name="config")
def main(cfg):
    _, task, _, _ = instantiate_from_cfg(cfg)

    mj_model = task.ref.mj_scene.mj_model
    mj_data = mujoco.MjData(mj_model)
    visualize_trajectory(mj_model, mj_data, task.ref.time, task.ref.x)

if __name__ == "__main__":
    main()