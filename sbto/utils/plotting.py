import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

def plot_state_control(
        time: Array,
        x_traj: Array,
        u_traj: Array,
        knots: Array,
        Nq: int,
        Nu: int,
        title_prefix="Trajectory"
        ):
    """
    Plots:
    - Figure 1: Base position (3) and velocity (3)
    - Figure 2: Joint positions and velocities
    - Figure 3: Controls with knots highlighted
    """
    x_traj, v_traj = np.split(x_traj, [Nq], axis=1)
    u_traj = np.asarray(u_traj)
    knots = np.asarray(knots)
    if len(knots.shape) == 1:
        knots = knots.reshape(-1, Nu)

    start, end = time[0], time[-1]
    Nknots = knots.shape[0]
    t_knots = np.linspace(start, end, Nknots, endpoint=True)

    # Extract components
    base_pos = x_traj[:, 0:3]
    base_vel = v_traj[:, :3]
    base_w = v_traj[:, 3:6]
    joint_pos = x_traj[:, -Nu:]
    joint_vel = v_traj[:, -Nu:]

    # ---------------- FIGURE 1: Base ----------------
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig1.suptitle(f"{title_prefix} - Base States")
    labels_base = ['x', 'y', 'z']

    # Base position
    for i in range(3):
        axs1[0].plot(time, base_pos[:, i], label=f"Pos {labels_base[i]}")
    axs1[0].set_ylabel("Position [m]")
    axs1[0].legend()
    axs1[0].grid(True)

    # Base velocity
    for i in range(3):
        axs1[1].plot(time, base_vel[:, i], label=f"v {labels_base[i]}")
    axs1[1].set_xlabel("Time step")
    axs1[1].set_ylabel("Velocity [m/s]")
    axs1[1].legend()
    axs1[1].grid(True)

    # Base amgular velocity
    for i in range(3):
        axs1[2].plot(time, base_w[:, i], label=f"w {labels_base[i]}")
    axs1[2].set_xlabel("Time step")
    axs1[2].set_ylabel("Angular veloctiy [rad/s]")
    axs1[2].legend()
    axs1[2].grid(True)


    plt.tight_layout()

    # ---------------- FIGURE 2: Joints ----------------
    fig2, axs2 = plt.subplots(Nu, 2, figsize=(12, Nu * 2), sharex=True)
    fig2.suptitle(f"{title_prefix} - Joint States")

    if Nu == 1:
        axs2 = axs2[None, :]  # Handle single-joint case for consistent indexing

    for j in range(Nu):
        axs2[j, 0].plot(time, joint_pos[:, j], label=f"q[{j}]")
        axs2[j, 0].set_ylabel(f"q{j}")
        axs2[j, 0].grid(True)

        axs2[j, 1].plot(time, joint_vel[:, j], label=f"qd[{j}]", color='orange')
        axs2[j, 1].set_ylabel(f"qd{j}")
        axs2[j, 1].grid(True)

    axs2[-1, 0].set_xlabel("Time step")
    axs2[-1, 1].set_xlabel("Time step")

    plt.tight_layout()

    # ---------------- FIGURE 3: Controls ----------------
    fig3, axs3 = plt.subplots(Nu, 1, figsize=(10, 2 * Nu), sharex=True)
    fig3.suptitle(f"{title_prefix} - Controls")

    if Nu == 1:
        axs3 = [axs3]  # Handle single-control case

    if len(knots.shape) == 1:
        knots = knots.reshape(-1, Nu)
    
    for i in range(Nu):
        axs3[i].plot(time, u_traj[:, i], label=f"u[{i}]")
        axs3[i].scatter(t_knots, knots[:, i], color='red', marker='x', label="Knots")
        axs3[i].grid(True)
        axs3[i].legend()
        axs3[i].set_ylabel(f"u{i}")

    axs3[-1].set_xlabel("Time step")
    plt.tight_layout()

    plt.show()

def plot_costs(all_costs: Array, title: str = "Cost Distribution over Iterations"):
    """
    Plot the distribution of costs over optimization iterations.

    Args:
        all_costs (Array): Array of shape [Nit, N_samples] with
                           all sample costs at each iteration.
        title (str): Title for the plot.
    """
    all_costs = np.asarray(all_costs)
    Nit = all_costs.shape[0]

    plt.figure(figsize=(10, 5))

    # Boxplot per iteration
    plt.violinplot(
        all_costs.T,  # boxplot expects shape [N_samples, Nit]
        positions=np.arange(Nit),
        showmeans=True,
        showextrema=False,
    )

    # Overlay min cost curves
    min_cost = np.min(all_costs, axis=1)
    plt.plot(np.arange(Nit), min_cost, "o-", label="Min", color="tab:blue")

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()