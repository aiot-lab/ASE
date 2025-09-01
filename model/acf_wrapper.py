"""
ACF Wrapper Module

This module provides a unified interface to both the original and optimized
ACF implementations, allowing easy switching between them.

Features:
- ACF (Autocorrelation Function) calculation
- Motion statistics calculation with optimized performance
- Drop-in replacement for the original ACF class
- Significant performance improvements
"""

from model.optimized_acf import OptimizedACF
from model.ACF import ACF as OriginalACF
import sys
import os

# Fix import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import implementations


def get_acf(CFR, process_arg, use_optimized=True):
    """
    Get ACF instance using either the original or optimized implementation.

    The returned instance provides both ACF and motion statistics functionality,
    with identical interfaces but significantly improved performance when
    using the optimized implementation.

    Parameters:
    -----------
    CFR : numpy.ndarray
        CFR data to process with shape [freq, time, channel]
    process_arg : object
        Processing parameters, including:
        - windows_width: Width of the sliding window
        - step_size: Step size for the sliding window
        - topK: Number of top frequency bands to use
        - time_per_frame: Time per frame in seconds
        - num_topK_subcarriers (optional): Number of top subcarriers
    use_optimized : bool, optional
        Whether to use the optimized implementation (default: True)

    Returns:
    --------
    ACF instance
        Instance of either OriginalACF or OptimizedACF with:
        - acf property: Returns the ACF matrix
        - ms property: Returns the motion statistics matrix
    """
    if use_optimized:
        return OptimizedACF(CFR=CFR, process_arg=process_arg)
    else:
        return OriginalACF(CFR=CFR, process_arg=process_arg)


# Examples of usage for different scenarios

def example_usage():
    """
    Example of basic usage of the ACF wrapper.
    """
    import numpy as np
    from collections import namedtuple

    # Create sample data
    cfr_data = np.random.randn(64, 512, 2)  # [freq, time, channel]

    # Create process parameters
    ProcessArg = namedtuple('ProcessArg', [
        'windows_width', 'step_size', 'topK', 'time_per_frame', 'num_topK_subcarriers'
    ])

    process_arg = ProcessArg(
        windows_width=64,
        step_size=32,
        topK=64,
        time_per_frame=0.04,  # 40ms per frame
        num_topK_subcarriers=64
    )

    # Method 1: Using the optimized version directly
    from model.optimized_acf import OptimizedACF
    acf1 = OptimizedACF(CFR=cfr_data, process_arg=process_arg)

    # Method 2: Using the wrapper (recommended)
    acf2 = get_acf(CFR=cfr_data, process_arg=process_arg, use_optimized=True)

    # Access ACF and motion statistics
    acf_matrix = acf2.acf
    ms_matrix = acf2.ms

    print(f"ACF shape: {acf_matrix.shape}")
    print(f"MS shape: {ms_matrix.shape}")

    return acf_matrix, ms_matrix


def analyze_signal_example():
    """
    Example usage in analyze_signal.py context.
    """
    """
    # Original code:
    from model.ACF import ACF
    
    acf_merged = ACF(CFR=cfr_merged, process_arg=process_arg)
    acf_merged_matrix = acf_merged.acf
    acf_merged_ms = acf_merged.ms
    
    acf = ACF(CFR=cfr, process_arg=process_arg)
    acf_ms = acf.ms
    acf_matrix = acf.acf
    
    # Modified code using wrapper:
    from model.acf_wrapper import get_acf
    
    acf_merged = get_acf(CFR=cfr_merged, process_arg=process_arg, use_optimized=True)
    acf_merged_matrix = acf_merged.acf
    acf_merged_ms = acf_merged.ms
    
    acf = get_acf(CFR=cfr, process_arg=process_arg, use_optimized=True)
    acf_ms = acf.ms
    acf_matrix = acf.acf
    """
    pass  # Example code only
