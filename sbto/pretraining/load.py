import numpy as np

d = np.load("/Users/mustaphadaly/Desktop/sbto/sbto/data/rollout_time_x_u_obs_traj_scene_29dof_rl_format.npz")
print(d.keys())
print(d["actuator_pos"].shape)
print("pos min/max:", d["actuator_pos"].min(), d["actuator_pos"].max())