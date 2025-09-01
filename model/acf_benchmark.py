"""
ACF Performance Benchmark

This script benchmarks the performance of the original and optimized ACF implementations
and visualizes the results.
"""

from model.acf_wrapper import get_acf
from model.optimized_acf import OptimizedACF, acf_compare_performance
from model.ACF import ACF as OriginalACF
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import argparse

# Fix import paths
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import implementations


def generate_synthetic_cfr(num_freq=64, num_time=512, num_channels=2, seed=42):
    """Generate synthetic CFR data for testing."""
    np.random.seed(seed)
    return np.random.randn(num_freq, num_time, num_channels)


def benchmark_acf(cfr_data, process_arg, num_runs=3, test_ms=True):
    """Benchmark ACF implementations over multiple runs."""
    original_times = []
    optimized_times = []
    original_ms_times = []
    optimized_ms_times = []

    print(f"Running benchmark with {num_runs} iterations...")

    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")

        # Original implementation - ACF
        start_time = time.time()
        original_acf = OriginalACF(CFR=cfr_data, process_arg=process_arg)
        _ = original_acf.acf  # Force ACF calculation
        original_time = time.time() - start_time
        original_times.append(original_time)
        print(f"Original ACF implementation: {original_time:.3f} seconds")

        # Optimized implementation - ACF
        start_time = time.time()
        optimized_acf = OptimizedACF(CFR=cfr_data, process_arg=process_arg)
        _ = optimized_acf.acf  # Force ACF calculation
        optimized_time = time.time() - start_time
        optimized_times.append(optimized_time)
        print(f"Optimized ACF implementation: {optimized_time:.3f} seconds")

        if test_ms:
            # Original implementation - Motion Statistics
            start_time = time.time()
            _ = original_acf.ms  # Force MS calculation
            original_ms_time = time.time() - start_time
            original_ms_times.append(original_ms_time)
            print(
                f"Original MS implementation: {original_ms_time:.3f} seconds")

            # Optimized implementation - Motion Statistics
            start_time = time.time()
            _ = optimized_acf.ms  # Force MS calculation
            optimized_ms_time = time.time() - start_time
            optimized_ms_times.append(optimized_ms_time)
            print(
                f"Optimized MS implementation: {optimized_ms_time:.3f} seconds")

        # Verify results match on first run
        if i == 0:
            results_match = np.allclose(
                original_acf.acf, optimized_acf.acf, rtol=1e-5, atol=1e-5)
            print(f"ACF results match: {results_match}")

            if test_ms:
                ms_match = np.allclose(
                    original_acf.ms, optimized_acf.ms, rtol=1e-5, atol=1e-5)
                print(f"MS results match: {ms_match}")
                print(f"Original MS shape: {original_acf.ms.shape}")
                print(f"Optimized MS shape: {optimized_acf.ms.shape}")

    # Calculate average times
    avg_original = np.mean(original_times)
    avg_optimized = np.mean(optimized_times)
    speedup = avg_original / avg_optimized

    print(f"\nAverage original ACF time: {avg_original:.3f} seconds")
    print(f"Average optimized ACF time: {avg_optimized:.3f} seconds")
    print(f"Average ACF speedup: {speedup:.2f}x")

    results = {
        "original_times": original_times,
        "optimized_times": optimized_times,
        "avg_original": avg_original,
        "avg_optimized": avg_optimized,
        "speedup": speedup
    }

    if test_ms and original_ms_times and optimized_ms_times:
        avg_original_ms = np.mean(original_ms_times)
        avg_optimized_ms = np.mean(optimized_ms_times)
        ms_speedup = avg_original_ms / avg_optimized_ms

        print(f"\nAverage original MS time: {avg_original_ms:.3f} seconds")
        print(f"Average optimized MS time: {avg_optimized_ms:.3f} seconds")
        print(f"Average MS speedup: {ms_speedup:.2f}x")

        # Calculate combined speedup (for both ACF and MS)
        combined_original = avg_original + avg_original_ms
        combined_optimized = avg_optimized + avg_optimized_ms
        combined_speedup = combined_original / combined_optimized
        print(f"\nCombined speedup (ACF+MS): {combined_speedup:.2f}x")

        results.update({
            "original_ms_times": original_ms_times,
            "optimized_ms_times": optimized_ms_times,
            "avg_original_ms": avg_original_ms,
            "avg_optimized_ms": avg_optimized_ms,
            "ms_speedup": ms_speedup,
            "combined_speedup": combined_speedup
        })

    return results


def plot_benchmark_results(results):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))

    # ACF comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Original', 'Optimized'], [
            results['avg_original'], results['avg_optimized']])
    plt.title('Average ACF Execution Time')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels
    plt.text(0, results['avg_original'], f"{results['avg_original']:.3f}s",
             ha='center', va='bottom')
    plt.text(1, results['avg_optimized'], f"{results['avg_optimized']:.3f}s",
             ha='center', va='bottom')

    # Individual run comparison for ACF
    plt.subplot(2, 2, 2)
    runs = list(range(1, len(results['original_times']) + 1))
    plt.plot(runs, results['original_times'], 'o-', label='Original')
    plt.plot(runs, results['optimized_times'], 'o-', label='Optimized')
    plt.title('ACF Execution Time by Run')
    plt.xlabel('Run')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # MS comparison if available
    if 'avg_original_ms' in results and 'avg_optimized_ms' in results:
        plt.subplot(2, 2, 3)
        plt.bar(['Original', 'Optimized'], [
                results['avg_original_ms'], results['avg_optimized_ms']])
        plt.title('Average MS Execution Time')
        plt.ylabel('Time (seconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add text labels
        plt.text(0, results['avg_original_ms'], f"{results['avg_original_ms']:.3f}s",
                 ha='center', va='bottom')
        plt.text(1, results['avg_optimized_ms'], f"{results['avg_optimized_ms']:.3f}s",
                 ha='center', va='bottom')

        # Individual run comparison for MS
        plt.subplot(2, 2, 4)
        plt.plot(runs, results['original_ms_times'], 'o-', label='Original')
        plt.plot(runs, results['optimized_ms_times'], 'o-', label='Optimized')
        plt.title('MS Execution Time by Run')
        plt.xlabel('Run')
        plt.ylabel('Time (seconds)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        title = f'Performance Comparison\nACF Speedup: {results["speedup"]:.2f}x, MS Speedup: {results["ms_speedup"]:.2f}x\nCombined Speedup: {results["combined_speedup"]:.2f}x'
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f'ACF Performance Comparison\nSpeedup: {results["speedup"]:.2f}x',
                     fontsize=16)

    plt.tight_layout()

    # Save figure
    plt.savefig('acf_benchmark_results.png')
    print("Benchmark results saved to acf_benchmark_results.png")

    # Show figure
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ACF implementations")
    parser.add_argument("--num_freq", type=int, default=64,
                        help="Number of frequency bands")
    parser.add_argument("--num_time", type=int, default=512,
                        help="Number of time frames")
    parser.add_argument("--num_channels", type=int,
                        default=2, help="Number of channels")
    parser.add_argument("--windows_width", type=int,
                        default=64, help="Window width")
    parser.add_argument("--step_size", type=int, default=32, help="Step size")
    parser.add_argument("--topK", type=int, default=None,
                        help="Top K frequency bands")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of benchmark runs")
    parser.add_argument("--skip_ms", action="store_true",
                        help="Skip motion statistics benchmarking")
    args = parser.parse_args()

    # Set topK to num_freq if not specified
    if args.topK is None:
        args.topK = args.num_freq

    print(
        f"Generating synthetic CFR data with shape ({args.num_freq}, {args.num_time}, {args.num_channels})...")
    cfr_data = generate_synthetic_cfr(
        num_freq=args.num_freq,
        num_time=args.num_time,
        num_channels=args.num_channels
    )

    # Create process_arg with parameters
    ProcessArg = namedtuple('ProcessArg', [
        'windows_width', 'step_size', 'topK', 'time_per_frame', 'num_topK_subcarriers'
    ])

    process_arg = ProcessArg(
        windows_width=args.windows_width,
        step_size=args.step_size,
        topK=args.topK,
        time_per_frame=0.04,  # 40ms per frame
        num_topK_subcarriers=args.topK
    )

    # Run benchmark
    results = benchmark_acf(cfr_data, process_arg,
                            args.num_runs, not args.skip_ms)

    # Plot results
    plot_benchmark_results(results)

    # Demonstrate usage with wrapper
    print("\nDemonstrating usage with wrapper function...")

    start_time = time.time()
    acf = get_acf(CFR=cfr_data, process_arg=process_arg,
                  use_optimized=True)
    wrapper_time = time.time() - start_time

    print(
        f"Wrapper function with optimized implementation: {wrapper_time:.3f} seconds")
    print(f"ACF shape: {acf.acf.shape}")
    print(f"MS shape: {acf.ms.shape}")


if __name__ == "__main__":
    main()
