"""
Example of using the low-level backend integration in the Prism framework.

This example demonstrates:
1. Using the low-level backend registry
2. Executing operations with different backends
3. Automatic backend selection based on hardware
4. Performance comparison between backends
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Prism framework components
from src.core.lowlevel import (
    registry,
    get_optimal_backend_for_operation,
    execute_operation,
    vector_add,
    matrix_multiply
)


def benchmark_vector_addition():
    """
    Benchmark vector addition performance across different backends.
    """
    print("\n=== Vector Addition Benchmark ===")
    
    # Define vector sizes to test
    sizes = [1000, 10000, 100000, 1000000]
    
    # Available backends
    backends = ["assembly", "c", "numpy"]
    
    # Results dictionary
    results = {backend: [] for backend in backends}
    
    # Run benchmarks
    for size in sizes:
        print(f"\nTesting vector size: {size}")
        
        # Create test data
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Reference result for verification
        expected = a + b
        
        # Benchmark each backend
        for backend in backends:
            times = []
            
            # Skip C backend for now as it's a placeholder
            if backend == "c":
                results[backend].append(0)
                continue
            
            for _ in range(5):  # Run multiple times for more stable results
                if backend == "numpy":
                    # Benchmark NumPy
                    start = time.time()
                    result = a + b
                    end = time.time()
                else:
                    # Benchmark Prism backend
                    start = time.time()
                    result = vector_add(a, b, backend=backend)
                    end = time.time()
                
                elapsed = (end - start) * 1000  # Convert to milliseconds
                times.append(elapsed)
                
                # Verify result
                if not np.allclose(result, expected, rtol=1e-5, atol=1e-5):
                    print(f"Warning: {backend} backend produced incorrect results!")
            
            # Record average time
            avg_time = sum(times) / len(times)
            results[backend].append(avg_time)
            print(f"  {backend}: {avg_time:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for backend in backends:
        if backend == "c":
            continue  # Skip C backend for now
        plt.plot(sizes, results[backend], marker='o', label=backend)
    
    plt.title("Vector Addition Performance")
    plt.xlabel("Vector Size")
    plt.ylabel("Execution Time (ms)")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig("vector_addition_benchmark.png")
    print(f"\nBenchmark plot saved to vector_addition_benchmark.png")


def benchmark_matrix_multiplication():
    """
    Benchmark matrix multiplication performance across different backends.
    """
    print("\n=== Matrix Multiplication Benchmark ===")
    
    # Define matrix sizes to test
    sizes = [(10, 10), (50, 50), (100, 100), (200, 200)]
    
    # Available backends
    backends = ["assembly", "c", "numpy"]
    
    # Results dictionary
    results = {backend: [] for backend in backends}
    
    # Run benchmarks
    for size in sizes:
        print(f"\nTesting matrix size: {size}")
        
        # Create test data
        a = np.random.rand(*size).astype(np.float32)
        b = np.random.rand(*size).astype(np.float32)
        
        # Reference result for verification
        expected = np.matmul(a, b)
        
        # Benchmark each backend
        for backend in backends:
            times = []
            
            # Skip C backend for now as it's a placeholder
            if backend == "c":
                results[backend].append(0)
                continue
            
            for _ in range(3):  # Run multiple times for more stable results
                if backend == "numpy":
                    # Benchmark NumPy
                    start = time.time()
                    result = np.matmul(a, b)
                    end = time.time()
                else:
                    # Benchmark Prism backend
                    start = time.time()
                    result = matrix_multiply(a, b, backend=backend)
                    end = time.time()
                
                elapsed = (end - start) * 1000  # Convert to milliseconds
                times.append(elapsed)
                
                # Verify result
                if not np.allclose(result, expected, rtol=1e-4, atol=1e-4):
                    print(f"Warning: {backend} backend produced incorrect results!")
            
            # Record average time
            avg_time = sum(times) / len(times)
            results[backend].append(avg_time)
            print(f"  {backend}: {avg_time:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for backend in backends:
        if backend == "c":
            continue  # Skip C backend for now
        plt.plot([s[0] for s in sizes], results[backend], marker='o', label=backend)
    
    plt.title("Matrix Multiplication Performance")
    plt.xlabel("Matrix Size (N×N)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig("matrix_multiplication_benchmark.png")
    print(f"\nBenchmark plot saved to matrix_multiplication_benchmark.png")


def demonstrate_automatic_backend_selection():
    """
    Demonstrate automatic backend selection based on hardware capabilities.
    """
    print("\n=== Automatic Backend Selection ===")
    
    # Create test data
    size = 10000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Get optimal backend for vector addition
    optimal_backend = get_optimal_backend_for_operation("vector_add")
    print(f"Optimal backend for vector addition: {optimal_backend}")
    
    # Execute operation without specifying backend (uses optimal)
    start = time.time()
    result = vector_add(a, b)
    end = time.time()
    
    print(f"Time with automatic selection: {(end - start) * 1000:.2f} ms")
    
    # Compare with explicit numpy
    start = time.time()
    numpy_result = a + b
    end = time.time()
    
    print(f"Time with numpy: {(end - start) * 1000:.2f} ms")
    print(f"Results match: {np.allclose(result, numpy_result, rtol=1e-5, atol=1e-5)}")


def main():
    """Main function to run all demonstrations."""
    # Demonstrate automatic backend selection
    demonstrate_automatic_backend_selection()
    
    # Run benchmarks
    benchmark_vector_addition()
    benchmark_matrix_multiplication()


if __name__ == "__main__":
    main()