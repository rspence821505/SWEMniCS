import numpy as np
from data_io import save_pickle, load_pickle


# write rmse function
def rmse(true_signal, analysis_signal):
    """
    Calculate the Root Mean Square Error (RMSE) between the true signal and the analysis signal.

    Parameters:
    -----------
    true_signal : np.ndarray
        The true signal values.
    analysis_signal : np.ndarray
        The analysis signal values.

    Returns:
    --------
    rmse : float
        The calculated RMSE value.
    """
    return np.sqrt(np.mean((true_signal - analysis_signal) ** 2))


def data_misfit(true_signal, analysis_signal, H, obs_time_indices=None):
    """
    Calculate the relative misfit between the true signal and the analysis signal.

    Parameters:
    -----------
    true_signal : np.ndarray
        The true signal values.
    analysis_signal : np.ndarray
        The analysis signal values.
    obs_time_indices : np.ndarray
        The observation time indices.
    H : np.ndarray
        The observation operator.

    Returns:
    --------
    relative_misfit : float
        The calculated relative misfit value.
    """
    if obs_time_indices is not None:
        true_obs = H @ true_signal[obs_time_indices].T
        pred_obs = H @ analysis_signal[obs_time_indices].T

    else:
        true_obs = H @ true_signal
        pred_obs = H @ analysis_signal

    return np.linalg.norm(true_obs - pred_obs, ord=2)


# write relative misfit function
def relative_misfit(true_signal, analysis_signal, H, obs_time_indices=None):
    """
    Calculate the relative misfit between the true signal and the analysis signal.

    Parameters:
    -----------
    true_signal : np.ndarray
        The true signal values.
    analysis_signal : np.ndarray
        The analysis signal values.
    obs_time_indices : np.ndarray
        The observation time indices.
    H : np.ndarray
        The observation operator.

    Returns:
    --------
    relative_misfit : float
        The calculated relative misfit value.
    """
    if obs_time_indices is not None:
        true_obs = H @ true_signal[obs_time_indices].T
        pred_obs = H @ analysis_signal[obs_time_indices].T

    else:
        true_obs = H @ true_signal
        pred_obs = H @ analysis_signal

    return np.linalg.norm(true_obs - pred_obs, ord=2) / np.linalg.norm(true_obs, ord=2)


def calculate_analysis_metrics(
    analysis_name,
    analysis_data=None,
    filename=None,
    save_first=False,
    display_name=None,
):
    """
    Calculate RMSE and misfit metrics for analysis data.

    Parameters:
    -----------
    analysis_name : str
        Name of the analysis (used for filename if not provided)
    analysis_data : numpy.ndarray, optional
        The analysis data. If provided and save_first=True, will save to file first
    filename : str, optional
        Pickle filename. If not provided, uses '{analysis_name}_analysis.pkl'
    save_first : bool, default=False
        If True, saves analysis_data to file before loading (useful for DCI WME case)
    display_name : str, optional
        Name to display in print output. If not provided, uses analysis_name

    Returns:
    --------
    tuple
        (rmse, misfit) values
    """

    # Set default filename if not provided
    if filename is None:
        filename = f"{analysis_name}_analysis.pkl"

    # Set default display name if not provided
    if display_name is None:
        display_name = analysis_name.upper()

    # Save analysis data first if requested (for DCI WME case)
    if save_first and analysis_data is not None:
        save_pickle(filename, analysis_data)

    # Load analysis data from pickle file
    analysis = load_pickle(filename)

    # Load result if not already provided
    result = load_pickle("setup_result.pkl")

    # Load true signal
    true_signal = load_pickle("true_signal.pkl")

    # Calculate metrics
    data_rmse = rmse(true_signal, analysis)  # time averaged RMSE
    data_relative_misfit = relative_misfit(
        true_signal, analysis, result["H"], result["obs_time_indices"]
    )  # relative misfit

    print(
        f"{display_name} RMSE: {data_rmse:<.10f}, {display_name}, Relative Misfit: {data_relative_misfit:<.4f}"
    )

    return data_rmse, data_relative_misfit
