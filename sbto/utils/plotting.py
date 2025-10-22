from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import os

Array = npt.NDArray[np.float64]

def plot_state_control(
        time: Array,
        x_traj: Array,
        u_traj: Array,
        knots: Array,
        Nq: int,
        Nu: int,
        title_prefix="Trajectory",
        save_dir: str=""
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

    if save_dir:
        if not os.path.exists(save_dir):
            Warning(f"Directory {save_dir} does not exists.")
            os.makedirs(save_dir)
        
        filenames = [
            "base_pos_vel",
            "joint_pos_vel",
            "controls_knots",
        ]
        format = "pdf"
        for fig, filename in zip(
            [fig1, fig2, fig3], filenames
        ):  
            filepath = os.path.join(save_dir, filename) + f'.{format}'
            fig.savefig(filepath, format=format)
            print("Figure saved to", filepath)
    else:
        plt.show()

def plot_costs(
        all_costs: Array,
        title: str = "Cost Distribution over Iterations",
        save_dir: str = "",
        ):
    """
    Plot the distribution of costs over optimization iterations.

    Args:
        all_costs (Array): Array of shape [Nit, N_samples] with
                           all sample costs at each iteration.
        title (str): Title for the plot.
    """
    all_costs = np.asarray(all_costs)
    # max cost is max cost at iteration 0 (remove outliers)
    max_lim_cost = 3. * np.mean(all_costs[0, :])
    all_costs = np.clip(all_costs, None, max_lim_cost)
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
    plt.ylim(top=max_lim_cost)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()


    if save_dir:
        if not os.path.exists(save_dir):
            Warning(f"Directory {save_dir} does not exists.")
            os.makedirs(save_dir)
        
        filename = "cost_over_iterations"
        format = "pdf"
        filepath = os.path.join(save_dir, filename) + f".{format}"
        plt.savefig(fname=filepath, format=format)
        print("Figure saved to", filepath)

    # When running on the server
    else:
        plt.show()


def plot_mean_cov(
    time,
    mean_knots, 
    knots, 
    cov,
    u_traj,
    Nu: int,
    save_dir: str = ""
    ):
    """
    Plot the mean, variance (diagonal of covariance), and best sample per control dimension.

    Args:
        time: (T) time array
        mean_knots: (Nknots, Nu) array of mean controls over time
        knots: (Nknots, Nu) array of best (elite) controls over time
        cov: (Nknots, Nu, Nu) full covariance matrices (only diagonals are used for plotting)
        u_traj: (T, Nu) full pd target trajectory
        Nu: number of control dimensions
        save_dir: Save direectory path
    """
    # ---------------- FIGURE: Control distribution ----------------
    fig, axs = plt.subplots(Nu, 1, figsize=(10, 2.5 * Nu), sharex=True)
    plt.title("Control Distribution", fontsize=14, fontweight="bold")

    # Handle single-control case
    if Nu == 1:
        axs = [axs]

    # Ensure correct shapes
    mean_knots = mean_knots.reshape(-1, Nu)
    knots = knots.reshape(-1, Nu)
    diag_cov = np.diag(cov).reshape(-1, Nu)

    T = mean_knots.shape[0]

    start, end = time[0], time[-1]
    Nknots = knots.shape[0]
    t_knots = np.linspace(start, end, Nknots, endpoint=True)

    # Plot each control dimension
    for i in range(Nu):
        ax = axs[i]
        mean = mean_knots[:, i]
        std = np.sqrt(diag_cov[:, i])

        ax.plot(time, u_traj[:, i], label=f"u[{i}]")

        ax.scatter(t_knots, mean, label=f"mean u[{i}]", color="C0", marker='o',)
        ax.fill_between(t_knots, mean - std, mean + std, color="C0", alpha=0.3, label="±1σ")
        ax.scatter(t_knots, knots[:, i], label="best", color="C1", marker='x',)

        ax.set_ylabel(f"$u_{i}$")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axs[-1].set_xlabel("Time step")

    plt.tight_layout()

    if save_dir:
        if not os.path.exists(save_dir):
            Warning(f"Directory {save_dir} does not exists.")
            os.makedirs(save_dir)
        
        filename = "control_knots_final_distrib"
        format = "pdf"
        filepath = os.path.join(save_dir, filename) + f".{format}"
        plt.savefig(fname=filepath, format=format)
        print("Figure saved to", filepath)

    # When running on the server
    else:
        plt.show()

def plot_contact_plan(
    contact_array: Array,
    ref_array: Optional[Array] = None,
    ee_labels: List[str] = [],
    dt: float = 0.01,
    size: int = 15,
    save_dir: str = ""
    ):
    """
    Visualize quadruped contact plan with optional reference.

    Args:
        contact_array (np.ndarray): shape [T, N_eeff], 0/1 realized contact status.
        ref_array (np.ndarray): optional reference contact plan, same shape.
        ee_labels (list of str): optional names for end-effectors.
        dt (float): 
    """
    T, N = contact_array.shape
    if len(ee_labels) == 0:
        ee_labels = [f"EE {i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, max(2, N * 0.6)))

    def draw_contacts(array, color, alpha=1.0, zorder=1, height=0.6):
        for i in range(N):
            y = N - 1 - i
            in_contact = array[:, i]
            starts = np.where(np.diff(np.pad(in_contact, (1, 0))) == 1)[0] * dt
            ends = np.where(np.diff(np.pad(in_contact, (0, 1))) == -1)[0] * dt
            for s, e in zip(starts, ends):
                ax.barh(y, e - s, left=s, height=height,
                        color=color, alpha=alpha, zorder=zorder)

    # Draw reference first (background)
    if ref_array is not None:
        draw_contacts(ref_array, color="lightgray", alpha=1.0, zorder=1, height=0.8)

    # Draw realized on top (foreground)
    draw_contacts(contact_array, color="dimgray", alpha=1.0, zorder=2, height=0.4)

    ax.set_xlim(0, T * dt)
    ax.set_ylim(-0.5, N - 0.5)
    ax.tick_params(axis='y', labelsize=size-2)
    ax.tick_params(axis='x', labelsize=size-2)
    ax.set_yticks(range(N))
    ax.set_yticklabels(labels=ee_labels[::-1])
    ax.set_xlabel("Time (s)",  size=size)
    ax.set_title("Contact Sequence (reference vs realized)", size=size+2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_dir:
        if not os.path.exists(save_dir):
            Warning(f"Directory {save_dir} does not exists.")
            os.makedirs(save_dir)
        
        filename = "contact_achieved_vs_planed"
        format = "pdf"
        filepath = os.path.join(save_dir, filename) + f".{format}"
        plt.savefig(fname=filepath, format=format)
        print("Figure saved to", filepath)

    else:
        plt.show()