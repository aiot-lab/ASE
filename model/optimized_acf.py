"""
Optimized ACF (Autocorrelation Function) implementation.

This module provides efficient implementations of ACF calculation,
including vectorized operations and Numba acceleration for maximum performance.
"""

import numpy as np
import numba
import os
import sys

# Fix import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


@numba.njit
def numba_correlate(x, y, mode="full"):
    """
    Numba-compatible implementation of signal.correlate
    """
    if mode == "full":
        n = len(x) + len(y) - 1
        result = np.zeros(n)
        for i in range(n):
            for j in range(len(x)):
                if i - j >= 0 and i - j < len(y):
                    result[i] += x[j] * y[i - j]
        return result
    else:
        raise ValueError("Only 'full' mode is supported for numba_correlate")


@numba.njit(parallel=True)
def batch_autocorrelation(cfr_windows, batch_size=10, compute_ms=False):
    """
    Calculate autocorrelation for multiple windows in parallel using Numba.

    Parameters:
    -----------
    cfr_windows : numpy.ndarray
        Array of CFR windows with shape [n_windows, window_width]
    batch_size : int
        Number of windows to process in each batch
    compute_ms : bool, optional
        If True, also return motion statistics (normalized ACF at lag=1)

    Returns:
    --------
    tuple
        Tuple of (autocorrelation results, motion statistics)
        When compute_ms=False, the second element is an empty array
    """
    n_windows = cfr_windows.shape[0]
    window_width = cfr_windows.shape[1]
    result = np.zeros((n_windows, window_width))

    # Always prepare an array for motion statistics
    ms_result = np.zeros(n_windows)

    for i in numba.prange(n_windows):
        # Get window data
        cfr_f = cfr_windows[i]

        # Normalize
        cfr_mean = np.mean(cfr_f)
        cfr_centered = cfr_f - cfr_mean
        norm = np.linalg.norm(cfr_centered)
        if norm > 1e-10:  # Avoid division by zero
            cfr_f_normal = cfr_centered / norm
        else:
            cfr_f_normal = cfr_centered

        # Calculate autocorrelation using numba-compatible function
        acf = numba_correlate(cfr_f_normal, cfr_f_normal, mode="full")

        # Store second half (since autocorrelation is symmetric)
        acf_half = acf[acf.size // 2:]
        result[i] = acf_half

        # If computing motion statistics, store ACF at lag=1 (normalized)
        if compute_ms:
            # MS is the normalized ACF at lag=1
            # Note: ACF at lag=0 is always 1.0 for normalized data
            if len(acf_half) > 1:
                ms_result[i] = acf_half[1]  # lag=1
            else:
                ms_result[i] = 0.0

    # Always return a tuple with both arrays, even if ms_result is empty
    return result, ms_result


def vectorized_acf_calculation(cfr_data, topK, windows_width, windows_step, num_channels, compute_ms=False):
    """
    Vectorized implementation of ACF calculation.

    Parameters:
    -----------
    cfr_data : numpy.ndarray
        CFR data with shape [freq, time, channel]
    topK : int
        Number of top frequency bands to use
    windows_width : int
        Width of the sliding window
    windows_step : int
        Step size for the sliding window
    num_channels : int
        Number of channels
    compute_ms : bool, optional
        If True, also compute motion statistics from ACF

    Returns:
    --------
    numpy.ndarray or tuple
        If compute_ms is False: ACF matrix with shape [freq, acf_len, time, channel]
        If compute_ms is True: Tuple of (ACF matrix, motion statistics matrix with shape [freq, time, channel])
    """
    frames = cfr_data.shape[1]

    # Preallocate output arrays
    time_steps = (frames - windows_width) // windows_step + 1
    acf_matrix = np.zeros((topK, windows_width, time_steps, num_channels))

    # If computing motion statistics, prepare an array for it with shape [freq, time, channel]
    if compute_ms:
        ms_matrix = np.zeros((topK, time_steps, num_channels))

    # Process each channel
    for channel_idx in range(num_channels):
        print(f"Processing channel {channel_idx+1}/{num_channels}")
        # Use only top K frequency bands
        cfr_for_channel = cfr_data[:topK, :, channel_idx]

        # Process each time window
        for t_idx, t in enumerate(range(0, frames - windows_width + 1, windows_step)):
            # Extract all frequency bands for this time window at once
            # Shape: [topK, windows_width]
            cfr_windows = cfr_for_channel[:, t:t + windows_width]

            # Reshape to prepare for batch processing
            cfr_windows_reshaped = cfr_windows.reshape(topK, windows_width)

            # Process in batches with Numba acceleration
            acf_results, ms_results = batch_autocorrelation(
                cfr_windows_reshaped, compute_ms=compute_ms)

            # Store results
            acf_matrix[:, :, t_idx, channel_idx] = acf_results

            # If computing motion statistics, store MS results with shape [freq, time, channel]
            if compute_ms:
                ms_matrix[:, t_idx, channel_idx] = ms_results

    if compute_ms:
        return acf_matrix, ms_matrix
    else:
        return acf_matrix


def get_acf_and_ms_optimized(cfr_data, topK, windows_width, windows_step, num_channels):
    """
    Optimized standalone function for calculating both ACF and motion statistics in one pass.

    Parameters:
    -----------
    cfr_data : numpy.ndarray
        CFR data with shape [freq, time, channel]
    topK : int
        Number of top frequency bands to use
    windows_width : int
        Width of the sliding window
    windows_step : int
        Step size for the sliding window
    num_channels : int
        Number of channels

    Returns:
    --------
    tuple
        (ACF matrix with shape [freq, acf_len, time, channel],
         MS matrix with shape [freq, time, channel])
    """
    # Ensure topK doesn't exceed available frequency bands
    topK = min(topK, cfr_data.shape[0])

    # Perform vectorized calculation with motion statistics
    return vectorized_acf_calculation(
        cfr_data=cfr_data,
        topK=topK,
        windows_width=windows_width,
        windows_step=windows_step,
        num_channels=num_channels,
        compute_ms=True
    )


def get_acf_optimized(cfr_data, topK, windows_width, windows_step, num_channels):
    """
    Optimized standalone function for ACF calculation.

    Parameters:
    -----------
    cfr_data : numpy.ndarray
        CFR data with shape [freq, time, channel]
    topK : int
        Number of top frequency bands to use
    windows_width : int
        Width of the sliding window
    windows_step : int
        Step size for the sliding window
    num_channels : int
        Number of channels

    Returns:
    --------
    numpy.ndarray
        ACF matrix with shape [freq, acf_len, time, channel]
    """
    # Ensure topK doesn't exceed available frequency bands
    topK = min(topK, cfr_data.shape[0])

    # Perform vectorized calculation without motion statistics
    return vectorized_acf_calculation(
        cfr_data=cfr_data,
        topK=topK,
        windows_width=windows_width,
        windows_step=windows_step,
        num_channels=num_channels,
        compute_ms=False
    )


class OptimizedACF:
    """
    Optimized implementation of ACF calculation.

    This class provides a drop-in replacement for the ACF class with improved performance.
    """

    def __init__(self, CFR, process_arg):
        """
        Initialize the OptimizedACF calculator.

        Parameters:
        -----------
        CFR : numpy.ndarray
            CFR data with shape [freq, time, channel]
        process_arg : object
            Object containing processing parameters
        """
        self._CFR = CFR  # Set the internal _CFR attribute directly
        self.num_channels = CFR.shape[2] if len(CFR.shape) > 2 else 1
        self.frames = CFR.shape[1]

        # Extract parameters from process_arg
        self.windows_width = getattr(process_arg, "windows_width", 64)
        self.windows_step = getattr(process_arg, "step_size", 32)
        self.topK = getattr(process_arg, "topK", CFR.shape[0])
        self.time_per_frame = getattr(process_arg, "time_per_frame", 0.04)

        # For compatibility with original ACF implementation
        try:
            if hasattr(process_arg, "num_topK_subcarriers") and process_arg.num_topK_subcarriers > 0:
                self._topK = min(
                    process_arg.num_topK_subcarriers, self._CFR.shape[0])
            else:
                self._topK = self._CFR.shape[0]
        finally:
            assert self._topK <= self._CFR.shape[0], "topK should be less than the number of subcarriers"

        self._windows_width = self.windows_width
        self._windows_step = self.windows_step

        # Compute ACF and MS once in a single pass (caching the results)
        self._acf_result = None
        self._ms_result = None
        self._results_computed = False

    def _compute_results(self):
        """
        Compute both ACF and MS in a single pass and cache the results.
        """
        if not self._results_computed:
            self._acf_result, self._ms_result = get_acf_and_ms_optimized(
                cfr_data=self._CFR,
                topK=self.topK,
                windows_width=self.windows_width,
                windows_step=self.windows_step,
                num_channels=self.num_channels
            )
            self._results_computed = True

    @property
    def acf(self):
        """
        Get the ACF matrix.

        Returns:
        --------
        numpy.ndarray
            ACF matrix with shape [freq, acf_len, time, channel]
        """
        self._compute_results()
        return self._acf_result

    @property
    def ms(self):
        """
        Get the motion statistics matrix.

        Returns:
        --------
        numpy.ndarray
            Motion statistics matrix with shape [freq, time, channel]
        """
        self._compute_results()
        return self._ms_result

    @property
    def CFR(self):
        return self._CFR

    @CFR.setter
    def CFR(self, value):
        self._CFR = value
        self._results_computed = False  # Reset cache when CFR changes

    @property
    def topK(self):
        return self._topK

    @topK.setter
    def topK(self, value):
        self._topK = value
        self._results_computed = False  # Reset cache when topK changes

    @property
    def windows_width(self):
        return self._windows_width

    @windows_width.setter
    def windows_width(self, value):
        self._windows_width = value
        self._results_computed = False  # Reset cache when windows_width changes

    @property
    def windows_step(self):
        return self._windows_step

    @windows_step.setter
    def windows_step(self, value):
        self._windows_step = value
        self._results_computed = False  # Reset cache when windows_step changes

    @staticmethod
    def get_motion_curve(ms):
        """
        Get the motion curve by averaging across frequencies.

        Parameters:
        -----------
        ms : numpy.ndarray
            Motion statistics matrix

        Returns:
        --------
        numpy.ndarray
            Motion curve
        """
        return np.mean(ms, axis=1)


def acf_compare_performance(cfr_data, process_arg):
    """
    Compare performance between original and optimized ACF implementations.

    Parameters:
    -----------
    cfr_data : numpy.ndarray
        CFR data to process
    process_arg : object
        Processing parameters

    Returns:
    --------
    tuple
        (original_result, optimized_result, speedup_factor)
    """
    import time
    from model.ACF import ACF as OriginalACF

    # Test original implementation
    print("Running original implementation...")
    start_time = time.time()
    original_acf = OriginalACF(CFR=cfr_data, process_arg=process_arg)
    original_time = time.time() - start_time
    print(f"Original implementation: {original_time:.3f} seconds")

    # Test optimized implementation
    print("Running optimized implementation...")
    start_time = time.time()
    optimized_acf = OptimizedACF(CFR=cfr_data, process_arg=process_arg)
    optimized_time = time.time() - start_time
    print(f"Optimized implementation: {optimized_time:.3f} seconds")

    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")

    # Verify results match
    results_match = np.allclose(
        original_acf.acf, optimized_acf.acf, rtol=1e-5, atol=1e-5)
    print(f"ACF results match: {results_match}")

    ms_match = np.allclose(
        original_acf.ms, optimized_acf.ms, rtol=1e-5, atol=1e-5)
    print(f"MS results match: {ms_match}")

    return original_acf.acf, optimized_acf.acf, speedup
