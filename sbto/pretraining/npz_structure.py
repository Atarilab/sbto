import numpy as np

npz_path = "sbto/data/rollout_time_x_u_obs_traj.npz"

data = np.load(npz_path)

print("KEYS AND SHAPES ")
for key in data.files:
    arr = data[key]
    print(f"{key:25s} shape = {arr.shape}")