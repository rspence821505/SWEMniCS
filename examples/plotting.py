import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
                (true_signal.vals[:, :, 0] - hb)[obs_indices, station_idx],
                (analysis[:, :, 0] - hb)[obs_indices, station_idx],
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
            y_data=[true_signal.vals[obs_indices, 0, 0], analysis[obs_indices, 0, 0]],
            styles=["solid", "dashed"],
            colors=["blue", "red"],
            labels=["True", "Analysis"],
            title="Tidal Height at 800 m for SUPG Scheme",
            xlabel="Time (days)",
            ylabel="Height (m)",
            filename=f"{save_prefix}tidal_height_SUPG.png" if save else None,
        )

        # Plot 3: Tidal Velocity
        _create_plot(
            x_data=plot_grid,
            y_data=[true_signal.vals[obs_indices, 0, 1]],
            styles=["solid"],
            colors=["blue"],
            labels=["True"],
            title="Tidal Velocity at 800 m for SUPG Scheme",
            xlabel="Time (Days)",
            ylabel="Velocity (m/s)",
            filename=f"{save_prefix}tidal_velocity_SUPG.png" if save else None,
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
