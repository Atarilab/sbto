import numpy as np
import csv
import os

npz_path = "sbto/data/time_x_u_traj_rl_format.npz"
out_csv = "sbto/data/time_x_u_traj_rl_format_full.csv"


data = np.load(npz_path)

print("=== Building single CSV ===")
print("Input:", npz_path)
print("Output:", out_csv)


first_key = list(data.files)[0]
T = len(data[first_key])
print("Detected T =", T)

flat_columns = []
header = []

for key in data.files:
    arr = data[key]

    if arr.ndim == 1:
        flat = arr.reshape(T, 1)

        # header names
        col_names = [f"{key}"]

    elif arr.ndim == 2:
        # shape (T, D)
        flat = arr

        col_names = [f"{key}_{i}" for i in range(arr.shape[1])]

    elif arr.ndim == 3:
        # shape (T, D1, D2)
        flat = arr.reshape(T, arr.shape[1] * arr.shape[2])

        col_names = [f"{key}_{i}" for i in range(flat.shape[1])]

    else:
        print(f"Skipping {key}: unsupported ndim={arr.ndim}")
        continue

    flat_columns.append(flat)
    header.extend(col_names)

full_matrix = np.concatenate(flat_columns, axis=1)
print("Full matrix shape:", full_matrix.shape)

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(full_matrix)

print("\nDONE!")
print("CSV saved to:", out_csv)