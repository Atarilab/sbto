import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_columns_prefix(df, prefix: str):
    """
    Return all columns in the dataframe that represent error metrics.
    """
    return [c for c in df.columns if c.startswith(prefix)]

def get_error_columns(df):
    return get_columns_prefix(df, "err_")

def get_act_acc_columns(df):
    return get_columns_prefix(df, "act_acc")

def plot_histograms_columns(df, columns, bins=50, figsize=(6,4)):
    """
    Plot seaborn histograms for all error columns.
    One figure per error metric.

    Parameters
    ----------
    df : pd.DataFrame
    bins : int
        Number of histogram bins.
    figsize : tuple
        Size of each figure.
    """
    for col in columns:
        plt.figure(figsize=figsize)
        sns.histplot(df[col].dropna(), bins=bins, kde=False)
        mean = np.mean(df[col].dropna())
        std = np.std(df[col].dropna())
        plt.title(f"Histogram of {col} (mean: {mean:.2f}, std: {std:.2f})")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

def plot_histograms_columns_grid(df, columns, bins=50, cols=3, figsize=(5,4)):
    """
    Plot all histograms in a grid layout (multipanel figure).

    Parameters
    ----------
    df : pd.DataFrame
    bins : int
        Number of bins per histogram.
    cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        Size of each subplot.
    """
    n = len(columns)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize[0], rows * figsize[1]))
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        sns.histplot(df[col].dropna(), bins=bins, kde=False, ax=ax)
        mean = np.mean(df[col].dropna())
        std = np.std(df[col].dropna())
        ax.set_title(f"{col} (mean: {mean:.2f}, std: {std:.2f})")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # turn off leftover axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_error_histograms(df, bins=50, figsize=(6,4)):
    columns = get_error_columns(df)
    return plot_histograms_columns(df, columns, bins=bins, figsize=figsize)

def plot_error_histograms_grid(df, bins=50, cols=3, figsize=(5,4)):
    columns = get_error_columns(df)
    return plot_histograms_columns_grid(df, columns, bins=bins, cols=cols, figsize=figsize)

def plot_act_acc_histograms_grid(df, bins=50, cols=3, figsize=(5,4)):
    columns = get_act_acc_columns(df)
    return plot_histograms_columns_grid(df, columns, bins=bins, cols=cols, figsize=figsize)

def plot_T_vs_duration(df, figsize=(6,4)):
    # plt.figure(figsize=figsize)
    sns.relplot(data=df, x="T", y="opt_duration")
    plt.title("Optimization time (s) vs. opt timesteps T")
    plt.tight_layout()
    plt.show()

def plot_cost_vs_opt_n_it(df, figsize=(6,4)):
    # plt.figure(figsize=figsize)
    sns.relplot(data=df, x="min_cost", y="opt_n_it")
    plt.title("Cost vs. opt iterations")
    plt.tight_layout()
    plt.show()


def plot_histograms_columns_grid_compare(
    df_sbto,
    df_mpc,
    columns,
    bins=50,
    cols=3,
    figsize=(5, 4),
    alpha=0.6
):
    """
    Compare SBTO vs MPC error histograms in a grid layout.

    Parameters
    ----------
    df_sbto : pd.DataFrame
        SBTO dataset (algo == 'SBTO')
    df_mpc : pd.DataFrame
        MPC dataset (algo == 'SBMPC')
    bins : int
        Number of histogram bins.
    cols : int
        Number of columns in subplot grid.
    figsize : tuple
        Size of each subplot.
    alpha : float
        Histogram transparency.
    """
    n = len(columns)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize[0], rows * figsize[1]),
        sharex=False
    )
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        sbto_vals = df_sbto[col].dropna()
        mpc_vals = df_mpc[col].dropna()

        sns.histplot(
            sbto_vals,
            bins=bins,
            ax=ax,
            stat="density",
            element="step",
            fill=True,
            alpha=alpha,
            label="SBTO"
        )

        sns.histplot(
            mpc_vals,
            bins=bins,
            ax=ax,
            stat="density",
            element="step",
            fill=True,
            alpha=alpha,
            label="SBMPC"
        )

        sbto_mean, sbto_std = sbto_vals.mean(), sbto_vals.std()
        mpc_mean, mpc_std = mpc_vals.mean(), mpc_vals.std()

        ax.set_title(
            f"{col}\n"
            f"SBTO μ={sbto_mean:.2f}, σ={sbto_std:.2f} | "
            f"MPC μ={mpc_mean:.2f}, σ={mpc_std:.2f}"
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.legend(fontsize=8)

    # turn off unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_ecdf_columns_grid_compare(
    df_sbto,
    df_mpc,
    columns,
    cols=3,
    figsize=(5, 4)
):
    """
    Compare SBTO vs MPC using ECDF plots in a grid layout.

    Parameters
    ----------
    df_sbto : pd.DataFrame
        SBTO dataset
    df_mpc : pd.DataFrame
        MPC dataset
    columns : list of str
        Columns to plot
    cols : int
        Number of columns for subplot grid
    figsize : tuple
        Size of each subplot
    """
    n = len(columns)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * figsize[0], rows * figsize[1]),
        sharex=False
    )
    axes = axes.flatten()

    for ax, col in zip(axes, columns):
        sbto_vals = df_sbto[col].dropna()
        mpc_vals = df_mpc[col].dropna()

        # ECDF for SBTO
        sns.ecdfplot(
            sbto_vals,
            ax=ax,
            label="SBTO",
            linewidth=2
        )

        # ECDF for MPC
        sns.ecdfplot(
            mpc_vals,
            ax=ax,
            label="SBMPC",
            linewidth=2
        )

        # Compute means and stds for display
        sbto_mean, sbto_std = sbto_vals.mean(), sbto_vals.std()
        mpc_mean, mpc_std = mpc_vals.mean(), mpc_vals.std()

        ax.set_title(
            f"{col}\n"
            f"SBTO μ={sbto_mean:.2f}, σ={sbto_std:.2f} | "
            f"MPC μ={mpc_mean:.2f}, σ={mpc_std:.2f}",
            fontsize=10
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # disable unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def plot_error_histograms_grid_compare(
    df_sbto,
    df_mpc,
    bins=50,
    cols=3,
    figsize=(5, 4),
    alpha=0.6
):
    """
    Compare SBTO vs MPC error histograms in a grid layout.

    Parameters
    ----------
    df_sbto : pd.DataFrame
        SBTO dataset (algo == 'SBTO')
    df_mpc : pd.DataFrame
        MPC dataset (algo == 'SBMPC')
    bins : int
        Number of histogram bins.
    cols : int
        Number of columns in subplot grid.
    figsize : tuple
        Size of each subplot.
    alpha : float
        Histogram transparency.
    """
    columns = get_error_columns(df_sbto)
    plot_histograms_columns_grid_compare(df_sbto, df_mpc, columns, bins, cols, figsize, alpha)

def plot_error_ecdf_grid_compare(
    df_sbto,
    df_mpc,
    cols=3,
    figsize=(5, 4),
):
    """
    Compare SBTO vs MPC error histograms in a grid layout.

    Parameters
    ----------
    df_sbto : pd.DataFrame
        SBTO dataset (algo == 'SBTO')
    df_mpc : pd.DataFrame
        MPC dataset (algo == 'SBMPC')
    cols : int
        Number of columns in subplot grid.
    figsize : tuple
        Size of each subplot.
    """
    columns = get_error_columns(df_sbto)
    plot_ecdf_columns_grid_compare(df_sbto, df_mpc, columns, cols, figsize)

if __name__ == "__main__":
    from sbto.evaluation.load import load_dataset_with_errors
    import multiprocessing as mp

    DATASET_ROOT = "datasets/G1RobotObjRef20Knots"
    mp.set_start_method("spawn", force=True)
    df = load_dataset_with_errors(DATASET_ROOT, num_workers=60)
    # df = df.nsmallest(200, "min_cost")
    # single-figure histograms
    # plot_error_histograms(df)
    plot_T_vs_duration(df)
    plot_cost_vs_opt_n_it(df)
    plot_histograms_columns_grid(df, ["T"])
    plot_histograms_columns_grid(df, ["opt_duration"])
    plot_histograms_columns_grid(df, ["total_sim_timesteps"])


    # grid view of all histograms
    plot_act_acc_histograms_grid(df, cols=3)
    plot_error_histograms_grid(df, cols=3)
    