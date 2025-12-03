import numpy as np

npz_path = "sbto/data/pretraining_actor_input.npz"

data = np.load(npz_path)

print("KEYS AND SHAPES ")
for key in data.files:
    arr = data[key]
    print(f"{key:25s} shape = {arr.shape}")