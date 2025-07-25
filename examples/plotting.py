import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.tri import Triangulation
from dolfinx.fem import Function

# from dolfinx import fem as fe
from matplotlib import gridspec
import matplotlib.patheffects as path_effects
import seaborn as sns

sns.set_palette("bright")


def create_bathymetry(h_b_val, V):
    """Create bathymetry over a given FunctionSpace"""
    h_b = Function(V.sub(0).collapse()[0])
    # make shore line at 13800
    h_b.interpolate(lambda x: h_b_val / 13800 * (13800 - x[0]))
    return h_b


def plot_mixed_function(
    msh,
    V,
    u_array,
    component=0,
    title="Mixed function component",
    stations=None,
    cmap="cividis",
    show_mesh=True,
    station_style="halo",
    show_station_labels=False,
    save_path=None,
):
    """
    Matplotlib-based 2D visualization of a mixed function component from raw array of coefficients,
    with full-height colorbar and enhanced station visualization options.

    Parameters:
    - station_style: str, options: 'halo', 'concentric'
    - show_station_labels: bool, whether to show station labels
    - save_path: str or None, file path to save the plot (e.g., 'output.png', 'figure.pdf')
    """
    coords = msh.geometry.x[:, :2]

    # Collapse mixed space and extract subcomponent
    V_sub = V.sub(component)
    V_sub_c, sub_map = V_sub.collapse()
    u_sub = Function(V_sub_c)
    u_sub.x.array[:] = u_array[sub_map]

    # Triangle mesh connectivity
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    cells = msh.topology.connectivity(tdim, 0).array.reshape(-1, 3)

    # Interpolate function to mesh vertices
    dof_coords = V_sub_c.tabulate_dof_coordinates()[:, :2]
    dof_to_vertex_map = np.array(
        [np.argmin(np.linalg.norm(dof_coords - x, axis=1)) for x in coords]
    )
    values_at_vertices = u_sub.x.array[dof_to_vertex_map]
    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    # Set up figure with single plot area
    fig, ax = plt.subplots(figsize=(10, 8))

    # Main plot
    tpc = ax.tripcolor(triang, values_at_vertices, shading="gouraud", cmap=cmap)
    if show_mesh:
        ax.triplot(triang, color="white", linewidth=0.7, alpha=1.0)

    # set tick mark label sizes
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.set_xlabel("x", fontsize=22)
    ax.set_ylabel("y", fontsize=22)
    ax.set_aspect("equal")

    # Make the mesh fill the entire plot area
    ax.margins(0)
    ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())

    # Enhanced Station Visualization
    if stations is not None:
        stations = np.asarray(stations)
        if stations.shape[1] == 3:
            stations = stations[:, :2]

        if station_style == "halo":
            # Halo/Glow Effect (multiple layers)
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=120,
                c="darkgrey",
                alpha=0.6,
                marker="o",
                zorder=10,
            )  # Outer glow
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=60,
                c="red",
                alpha=0.8,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                zorder=11,
            )  # Main marker

        elif station_style == "concentric":
            # Multi-ring Concentric Circles
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=100,
                c="none",
                edgecolors="red",
                linewidths=2.5,
                alpha=0.7,
                zorder=10,
            )
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=55,
                c="red",
                edgecolors="white",
                linewidths=1.5,
                zorder=11,
            )
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=20,
                c="white",
                edgecolors="black",
                linewidths=1,
                zorder=12,
            )

        else:
            # Default: simple red circles
            ax.scatter(
                stations[:, 0],
                stations[:, 1],
                s=80,
                c="red",
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=10,
            )

        # Text Labels with Halos (if requested)
        if show_station_labels:
            for i, (x, y) in enumerate(stations):
                ax.text(
                    x,
                    y + 0.05,
                    f"S{i+1}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                    path_effects=[
                        path_effects.withStroke(linewidth=3, foreground="black")
                    ],
                    zorder=12,
                )

    # Full-height colorbar - automatically sized to match plot
    cbar = fig.colorbar(tpc, ax=ax, aspect=14, pad=0.04, shrink=0.48)
    cbar.set_label("Height (m)", fontsize=20, labelpad=10)

    plt.tight_layout()

    # Save plot if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_comparison1(
    msh,
    V,
    plot_triplets,
    component=0,
    titles=None,
    stations=None,
    cmap="viridis",
    diff_cmap="PuOr",
    selected_days=None,
    save_path=None,
    dpi=300,
    format="png",
    show=True,
):
    """
    Matplotlib-based vertical comparison of triplets of difference solutions from raw arrays.
    Each row shows (4D-Var | DC-4D-Var | Difference 3).

    Parameters:
    - msh: dolfinx.mesh.Mesh
    - V: dolfinx.fem.FunctionSpace (mixed space)
    - plot_triplets: list of tuple(np.ndarray, np.ndarray, np.ndarray) - all three arrays are differences
    - component: int, index of the subcomponent to plot
    - titles: list of titles for each subplot
    - stations: optional Nx3 or Nx2 array of station coordinates
    - cmap: str, matplotlib colormap (not used anymore, kept for compatibility)
    - diff_cmap: str, matplotlib colormap for all difference plots
    - selected_days: list of int or None, which days to plot (1-based indexing). If None, plots all days.
                    Example: [1, 3, 5] will plot only days 1, 3, and 5
    - save_path: str, path to save the figure (if None, figure is not saved)
    - dpi: int, resolution for saved figure (default: 300)
    - format: str, file format for saved figure (default: 'png')
    - show: bool, whether to display the figure (default: True)
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from dolfinx.fem import Function
    import numpy as np

    # Filter plot_triplets based on selected_days
    if selected_days is None:
        # Plot all days
        filtered_triplets = plot_triplets
        day_labels = list(range(1, len(plot_triplets) + 1))
    else:
        # Plot only selected days
        selected_days = list(selected_days)  # Ensure it's a list
        filtered_triplets = []
        day_labels = []

        for day_num in selected_days:
            if 1 <= day_num <= (len(plot_triplets) + 1):
                filtered_triplets.append(
                    plot_triplets[day_num - 1]
                )  # Convert to 0-based indexing
                day_labels.append(day_num)
            else:
                print(
                    f"Warning: Day {day_num} is out of range (available days: 1-{len(plot_triplets)})"
                )

    if not filtered_triplets:
        raise ValueError("No valid days selected for plotting")

    coords = msh.geometry.x[:, :2]
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    cells = msh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    n_rows = len(filtered_triplets)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    tpc_diff = None  # store for difference colorbar

    for i, (diff1_arr, diff2_arr, diff3_arr) in enumerate(filtered_triplets):
        for j, u_array in enumerate([diff1_arr, diff2_arr, diff3_arr]):
            ax = axes[i, j]
            V_sub = V.sub(component)
            V_sub_c, sub_map = V_sub.collapse()
            u_sub = Function(V_sub_c)
            # u_sub.x.array[:] = u_array[sub_map]
            u_sub.x.array[:] = u_array
            dof_coords = V_sub_c.tabulate_dof_coordinates()[:, :2]
            dof_to_vertex_map = np.array(
                [np.argmin(np.linalg.norm(dof_coords - x, axis=1)) for x in coords]
            )
            values_at_vertices = u_sub.x.array[dof_to_vertex_map]

            # Use difference colormap for all plots
            tpc = ax.tripcolor(
                triang, values_at_vertices, shading="gouraud", cmap=diff_cmap
            )

            # Store tripcolor object for colorbar
            tpc_diff = tpc

            ax.triplot(triang, color="k", linewidth=0.3, alpha=0.5)
            ax.set_aspect("equal")

            # Make the mesh fill the entire plot area
            ax.margins(0)
            ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
            ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())

            # Only set x-label on bottom row
            if i == n_rows - 1:
                ax.set_xlabel("x")

            # Only set y-label on leftmost column
            if j == 0:
                # ax.tick_params(axis='y', labelsize=18)
                ax.set_ylabel("y", fontsize=20)

            # Only add titles to the top row
            if i == 0:
                if titles:
                    idx = 3 * i + j
                    if idx < len(titles):
                        ax.set_title(titles[idx])
                    else:
                        labels = ["4D-Var", "DC 4D-Var", "DC-WME 4D-Var"]
                        ax.set_title(labels[j])
                else:
                    labels = ["4D-Var", "DC 4D-Var", "DC-WME 4D-Var"]
                    ax.set_title(labels[j])

            if stations is not None:
                stations = np.asarray(stations)
                if stations.shape[1] == 3:
                    stations = stations[:, :2]
                ax.plot(stations[:, 0], stations[:, 1], "ro", markersize=5)

    # Adjust spacing between subplots and make room for colorbar - minimal vertical spacing
    plt.subplots_adjust(hspace=0.02, wspace=0.1, bottom=0.07, left=0.1, top=0.98)

    # Add "Day n" labels to each row using the actual day numbers
    for i in range(n_rows):
        # Get the position of the leftmost subplot in this row to align with its y-label
        ax_pos = axes[i, 0].get_position()
        row_center = ax_pos.y0 + ax_pos.height / 2
        actual_day = day_labels[i]
        fig.text(
            0.02,
            row_center,
            f"Day {actual_day}",
            rotation=90,
            fontsize=22,
            ha="center",
            va="center",
            weight="bold",
        )

    # Create single colorbar for all difference plots
    cbar_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    fig.colorbar(tpc_diff, cax=cbar_ax, orientation="horizontal").set_label(
        "Difference (m)", fontsize=16
    )

    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format=format, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show the figure if requested
    if show:
        plt.show()


def plot_comparison2(
    msh,
    V,
    plot_triplets,
    component=0,
    titles=None,
    stations=None,
    cmap="viridis",
    diff_cmap="PuOr",
    save_path=None,
    dpi=300,
    format="png",
    show=True,
):
    """
    Matplotlib-based vertical comparison of triplets of mixed-space solutions from raw arrays.
    Each row shows (Truth | Estimate | Difference).
    Parameters:
    - msh: dolfinx.mesh.Mesh
    - V: dolfinx.fem.FunctionSpace (mixed space)
    - plot_triplets: list of tuple(np.ndarray, np.ndarray, np.ndarray)
    - component: int, index of the subcomponent to plot
    - titles: list of titles for each subplot
    - stations: optional Nx3 or Nx2 array of station coordinates
    - cmap: str, matplotlib colormap
    - save_path: str, path to save the figure (if None, figure is not saved)
    - dpi: int, resolution for saved figure (default: 300)
    - format: str, file format for saved figure (default: 'png')
    - show: bool, whether to display the figure (default: True)
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from dolfinx.fem import Function
    import numpy as np

    coords = msh.geometry.x[:, :2]
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    cells = msh.topology.connectivity(tdim, 0).array.reshape(-1, 3)
    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    n_rows = len(plot_triplets)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(15, 2.8 * n_rows),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    tpc_main = None  # store for main colorbar (truth/estimate)
    tpc_diff = None  # store for difference colorbar

    for i, (truth_arr, estimate_arr, diff_arr) in enumerate(plot_triplets):
        for j, u_array in enumerate([truth_arr, estimate_arr, diff_arr]):
            ax = axes[i, j]
            V_sub = V.sub(component)
            V_sub_c, sub_map = V_sub.collapse()
            u_sub = Function(V_sub_c)

            # u_sub.x.array[:] = u_array[sub_map]
            u_sub.x.array[:] = u_array
            dof_coords = V_sub_c.tabulate_dof_coordinates()[:, :2]
            dof_to_vertex_map = np.array(
                [np.argmin(np.linalg.norm(dof_coords - x, axis=1)) for x in coords]
            )
            values_at_vertices = u_sub.x.array[dof_to_vertex_map]

            # Use different colormap for difference plots
            current_cmap = diff_cmap if j == 2 else cmap
            tpc = ax.tripcolor(
                triang, values_at_vertices, shading="gouraud", cmap=current_cmap
            )

            # Store tripcolor objects for colorbars
            if j < 2:  # Truth or Estimate
                tpc_main = tpc
            else:  # Difference
                tpc_diff = tpc
            ax.triplot(triang, color="k", linewidth=0.3, alpha=0.5)
            ax.set_aspect("equal")

            # Make the mesh fill the entire plot area
            ax.margins(0)
            ax.set_xlim(coords[:, 0].min(), coords[:, 0].max())
            ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())

            # Only set x-label on bottom row
            if i == n_rows - 1:
                ax.set_xlabel("x")

            # Only set y-label on leftmost column
            if j == 0:
                ax.set_ylabel("y")

            # Only add titles to the top row
            if i == 0:
                if titles:
                    idx = 3 * i + j
                    if idx < len(titles):
                        ax.set_title(titles[idx])
                    else:
                        labels = ["Truth", "Estimate", "Difference"]
                        ax.set_title(labels[j])
                else:
                    labels = ["Truth", "Estimate", "Difference"]
                    ax.set_title(labels[j])

            if stations is not None:
                stations = np.asarray(stations)
                if stations.shape[1] == 3:
                    stations = stations[:, :2]
                ax.plot(stations[:, 0], stations[:, 1], "ro", markersize=5)

    # Adjust spacing between subplots and make room for colorbar - minimal vertical spacing
    plt.subplots_adjust(hspace=0.02, wspace=0.1, bottom=0.07, left=0.1, top=0.98)

    # Add "Day n" labels to each row (after subplot adjustment to get correct positions)
    for i in range(n_rows):
        # Get the position of the leftmost subplot in this row to align with its y-label
        ax_pos = axes[i, 0].get_position()
        row_center = ax_pos.y0 + ax_pos.height / 2
        fig.text(
            0.02,
            row_center,
            f"Day {i+1}",
            rotation=90,
            fontsize=18,
            ha="center",
            va="center",
            weight="bold",
        )

    # Create two colorbars: main (2/3 width) and difference (1/3 width)
    total_width = 0.6
    main_width = total_width * (2 / 3)
    diff_width = total_width * (1 / 3)
    gap = 0.02

    # Main colorbar (Truth/Estimate)
    main_cbar_ax = fig.add_axes([0.18, 0.01, main_width, 0.02])
    fig.colorbar(tpc_main, cax=main_cbar_ax, orientation="horizontal").set_label(
        "Height (m)", fontsize=16
    )

    # Difference colorbar
    diff_cbar_ax = fig.add_axes([0.26 + main_width + gap, 0.01, diff_width, 0.02])
    fig.colorbar(tpc_diff, cax=diff_cbar_ax, orientation="horizontal").set_label(
        "Difference (m)", fontsize=16
    )

    # Save the figure if save_path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, format=format, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show the figure if requested
    if show:
        plt.show()


def plot_simulation_results(
    true_signal,
    analysis,
    y_obs,
    hb,
    problem_params,
    obs_indices,
    plot_params,
    station_idx=15,
    save=False,
    save_prefix="",
):
    """
    Create standardized plots for simulation results

    """

    # Setup time grid for plotting
    plot_steps = (problem_params["num_steps"] * problem_params["num_windows"]) + 1
    plot_final_time = problem_params["t_final"] / (60 * 60 * 24)  # Convert to days
    plot_grid = np.linspace(0, plot_final_time, plot_steps)[obs_indices]

    # Apply plotting style
    with plt.rc_context(plot_params):
        # Plot 1: Water Surface Elevation
        _create_plot(
            x_data=plot_grid,
            y_data=[
                (true_signal.vals[:, :, 0])[obs_indices, station_idx],
                (analysis[:, :, 0])[obs_indices, station_idx],
                y_obs[:, station_idx],
            ],
            styles=["solid", "dashed", "o"],
            colors=["blue", "red", "black"],
            labels=["True", "Analysis", "Observed"],
            title=f"Water Surface Elevation at {station_idx*50} m for SUPG Scheme",
            xlabel="Time (days)",
            ylabel="Water Surface Elevation (m)",
            filename=f"{save_prefix}water_surface_elevation_SUPG.png" if save else None,
        )

        # Plot 2: Tidal Height
        _create_plot(
            x_data=plot_grid,
            y_data=[
                (true_signal.vals[:, :, 0])[obs_indices, station_idx] - hb,
                (analysis[:, :, 0])[obs_indices, station_idx] - hb,
                y_obs[:, station_idx],
            ],
            styles=["solid", "dashed", "o"],
            colors=["blue", "red", "black"],
            labels=["True", "Analysis", "Observed"],
            title="Tidal Height at 800 m for SUPG Scheme",
            xlabel="Time (days)",
            ylabel="Height (m)",
            filename=f"{save_prefix}tidal_height_SUPG.png" if save else None,
        )

        # Plot 3: Tidal Velocity U
        _create_plot(
            x_data=plot_grid,
            y_data=[true_signal.vals[obs_indices, 0, 1], analysis[obs_indices, 0, 1]],
            styles=["solid", "dashed"],
            colors=["blue", "red"],
            labels=["True", "Analysis"],
            title="Tidal Velocity u at 800 m for SUPG Scheme",
            xlabel="Time (Days)",
            ylabel="Velocity (m/s)",
            filename=f"{save_prefix}tidal_velocity_SUPG.png" if save else None,
        )
        # Plot 4: Tidal Velocity V
        _create_plot(
            x_data=plot_grid,
            y_data=[true_signal.vals[obs_indices, 0, 2], analysis[obs_indices, 0, 2]],
            styles=["solid", "dashed"],
            colors=["blue", "red"],
            labels=["True", "Analysis"],
            title="Tidal Velocity v at 800 m for SUPG Scheme",
            xlabel="Time (Days)",
            ylabel="Velocity (m/s)",
            filename=f"{save_prefix}tidal_velocity_v_SUPG.png" if save else None,
        )


def _create_plot(
    x_data,
    y_data,
    styles,
    colors,
    labels,
    title,
    xlabel,
    ylabel=None,
    filename=None,
    grid=True,
    legend_loc="upper left",
):
    """
    Helper function to create a standardized plot

    """

    # Plot each data series
    for data, style, color, label in zip(y_data, styles, colors, labels):
        if style == "o":
            plt.plot(x_data, data, style, color=color, label=label)
        else:
            plt.plot(x_data, data, color=color, linestyle=style, label=label)

    # Set plot properties
    plt.grid(grid)
    plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legend_loc)

    # Save or show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def create_comparison_figure(
    true_signal,
    analysis,
    y_obs,
    hb,
    problem_params,
    obs_indices,
    plot_params,
    station_indices=(15, 25, 35),
    save=False,
    save_prefix="",
):
    """
    Create a multi-panel comparison figure showing different stations

    """

    # Setup time grid for plotting
    plot_steps = (problem_params["num_steps"] * problem_params["num_windows"]) + 1
    plot_final_time = problem_params["t_final"] / (60 * 60 * 24)  # Convert to days
    plot_grid = np.linspace(0, plot_final_time, plot_steps)[obs_indices]

    # Create multi-panel figure
    with plt.rc_context(plot_params):
        fig, axes = plt.subplots(
            len(station_indices), 1, figsize=(12, 4 * len(station_indices))
        )

        for i, station_idx in enumerate(station_indices):
            ax = axes[i] if len(station_indices) > 1 else axes

            # Plot true signal
            ax.plot(
                plot_grid,
                (true_signal.vals[:, :, 0] - hb)[obs_indices, station_idx],
                color="blue",
                linestyle="solid",
                label="True",
            )

            # Plot analysis result
            ax.plot(
                plot_grid,
                (analysis[:, :, 0] - hb)[obs_indices, station_idx],
                color="red",
                linestyle="dashed",
                label="Analysis",
            )

            # Plot observations
            ax.plot(
                plot_grid,
                y_obs[:, station_idx],
                "o",
                color="black",
                markersize=4,
                label="Observed",
            )

            # Set plot properties
            ax.grid(True)
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Water Surface Elevation (m)")
            ax.set_title(f"Station at {station_idx*50} m")
            ax.legend(loc="upper left")

        plt.tight_layout()
        if save:
            plt.savefig(
                f"{save_prefix}multi_station_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()
        plt.close()
