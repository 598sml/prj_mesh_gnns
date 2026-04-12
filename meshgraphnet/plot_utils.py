import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def make_comparison_plot(
    mesh_pos,
    cells,
    true_field,
    pred_field,
    error_field,
    component_name,
    save_path,
    title_prefix_pred="Predicted",
    title_suffix="",
    figsize=(8, 10),
    dpi=120,
):
    """
    Save a 3-panel comparison: truth, prediction, error.
    Uses a vertical layout with smaller colorbars.
    """

    triang = mtri.Triangulation(
        mesh_pos[:, 0].cpu().numpy(),
        mesh_pos[:, 1].cpu().numpy(),
        cells.cpu().numpy(),
    )

    true_np = true_field.cpu().numpy()
    pred_np = pred_field.cpu().numpy()
    err_np = error_field.cpu().numpy()

    vmin = min(true_np.min(), pred_np.min())
    vmax = max(true_np.max(), pred_np.max())

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)

    im0 = axes[0].tripcolor(triang, true_np, shading="flat", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"True {component_name} {title_suffix}".strip())
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, fraction=0.035, pad=0.02)

    im1 = axes[1].tripcolor(triang, pred_np, shading="flat", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title_prefix_pred} {component_name} {title_suffix}".strip())
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, fraction=0.035, pad=0.02)

    im2 = axes[2].tripcolor(triang, err_np, shading="flat")
    axes[2].set_title(f"Error {component_name} {title_suffix}".strip())
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal", adjustable="box")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, fraction=0.035, pad=0.02)

    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)


def save_rmse_plot(
    rmse_values,
    save_path,
    xlabel,
    ylabel="Velocity RMSE",
    title="RMSE",
    figsize=(6, 4),
    dpi=120,
):
    """
    Save a simple RMSE-vs-index line plot.
    """
    plt.figure(figsize=figsize)
    plt.plot(range(len(rmse_values)), rmse_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()