"""
Unified example of using low-level backends in the Prism framework.

This example demonstrates:
1. Using the backend registry to discover available backends
2. Selecting optimal backends for different operations
3. Benchmarking performance across different backends
4. Using hardware-specific optimizations
"""

import os
import sys
import time
import platform
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Prism framework components
import src as prism
from src.backends import get_backend_registry
from src.core.lowlevel import vector_add, matrix_multiply


def detect_platform():
    """
    Detect the current platform and architecture.
    
    Returns:
        Tuple of (platform_name, architecture)
    """
    system = platform.system()
    machine = platform.machine().lower()
    
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "Apple Silicon", "arm64"
    elif machine in ("arm64", "aarch64"):
        return "ARM", "arm64"
    elif machine in ("x86_64", "amd64"):
        return "x86_64", "x86_64"
    else:
        return system, machine


def list_available_backends():
    """
    List all available backends in the registry.
    """
    print("\n=== Available Backends ===")
    
    registry = get_backend_registry()
    backends = registry.list_backends()
    
    if not backends:
        print("No backends available.")
        return
    
    # Get platform info
    platform_name, arch = detect_platform()
    print(f"Current platform: {platform_name} ({arch})")
    
    # List all backends
    print("\nAll registered backends:")
    for name in sorted(backends):
        try:
            capabilities = registry.get_capabilities(name)
            features = ", ".join(capabilities.get("features", []))
            arch_support = ", ".join(capabilities.get("architectures", []))
            description = capabilities.get("description", "No description")
            
            print(f"- {name}: {description}")
            print(f"  Supported architectures: {arch_support}")
            print(f"  Features: {features}")
        except Exception as e:
            print(f"- {name}: Error retrieving capabilities: {str(e)}")
    
    # Find backends for current architecture
    arch_backends = registry.find_backends_for_arch(arch)
    print(f"\nBackends optimized for {arch}:")
    for name in sorted(arch_backends):
        capabilities = registry.get_capabilities(name)
        description = capabilities.get("description", "No description")
        print(f"- {name}: {description}")
    
    # Find backends for specific operations
    vector_backends = registry.find_backends_for_operation("vector_add")
    print("\nBackends supporting vector addition:")
    for name in sorted(vector_backends):
        capabilities = registry.get_capabilities(name)
        description = capabilities.get("description", "No description")
        print(f"- {name}: {description}")
    
    matrix_backends = registry.find_backends_for_operation("matrix_multiply")
    print("\nBackends supporting matrix multiplication:")
    for name in sorted(matrix_backends):
        capabilities = registry.get_capabilities(name)
        description = capabilities.get("description", "No description")
        print(f"- {name}: {description}")


def benchmark_vector_addition():
    """
    Benchmark vector addition across all available backends.
    """
    print("\n=== Vector Addition Benchmark ===")
    
    # Get backends that support vector addition
    registry = get_backend_registry()
    backend_names = registry.find_backends_for_operation("vector_add")
    
    if not backend_names:
        print("No backends available for vector addition.")
        return
    
    # Add NumPy as a reference
    backends = list(backend_names) + ["numpy"]
    
    # Define vector sizes to test
    sizes = [10_000, 100_000, 1_000_000, 10_000_000]
    
    # Results dictionary
    results = {backend: [] for backend in backends}
    
    # Run benchmarks
    for size in sizes:
        print(f"\nTesting vector size: {size}")
        
        # Create test data
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Reference result for verification
        numpy_result = a + b
        
        # Benchmark each backend
        for backend in backends:
            if backend == "numpy":
                # NumPy reference
                start = time.time()
                result = a + b
                end = time.time()
            else:
                # Use the backend through our core operations
                try:
                    with prism.use_backend(backend):
                        start = time.time()
                        result = vector_add(a, b)
                        end = time.time()
                except Exception as e:
                    print(f"  {backend}: Error - {str(e)}")
                    results[backend].append(None)
                    continue
            
            elapsed = (end - start) * 1000  # Convert to milliseconds
            results[backend].append(elapsed)
            
            # Verify result
            if not np.allclose(result, numpy_result, rtol=1e-5, atol=1e-5):
                print(f"  Warning: {backend} backend produced incorrect results!")
                max_diff = np.max(np.abs(result - numpy_result))
                print(f"  Maximum difference: {max_diff}")
            
            print(f"  {backend}: {elapsed:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for backend in backends:
        # Skip backends with errors
        if any(t is None for t in results[backend]):
            continue
        plt.plot(sizes, results[backend], marker='o', label=backend)
    
    plt.title("Vector Addition Performance")
    plt.xlabel("Vector Size")
    plt.ylabel("Execution Time (ms)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig("vector_addition_benchmark.png")
    print(f"\nBenchmark plot saved to vector_addition_benchmark.png")


def benchmark_matrix_multiplication():
    """
    Benchmark matrix multiplication across all available backends.
    """
    print("\n=== Matrix Multiplication Benchmark ===")
    
    # Get backends that support matrix multiplication
    registry = get_backend_registry()
    backend_names = registry.find_backends_for_operation("matrix_multiply")
    
    if not backend_names:
        print("No backends available for matrix multiplication.")
        return
    
    # Add NumPy as a reference
    backends = list(backend_names) + ["numpy"]
    
    # Define matrix sizes to test
    sizes = [(100, 100), (250, 250), (500, 500), (1000, 1000)]
    
    # Results dictionary
    results = {backend: [] for backend in backends}
    
    # Run benchmarks
    for size in sizes:
        m, n = size
        k = m  # Square matrices for simplicity
        print(f"\nTesting matrix size: {m}×{k} * {k}×{n}")
        
        # Create test data
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # Reference result for verification
        numpy_result = np.matmul(a, b)
        
        # Benchmark each backend
        for backend in backends:
            if backend == "numpy":
                # NumPy reference
                start = time.time()
                result = np.matmul(a, b)
                end = time.time()
            else:
                # Use the backend through our core operations
                try:
                    with prism.use_backend(backend):
                        start = time.time()
                        result = matrix_multiply(a, b)
                        end = time.time()
                except Exception as e:
                    print(f"  {backend}: Error - {str(e)}")
                    results[backend].append(None)
                    continue
            
            elapsed = (end - start) * 1000  # Convert to milliseconds
            results[backend].append(elapsed)
            
            # Verify result
            if not np.allclose(result, numpy_result, rtol=1e-4, atol=1e-4):
                print(f"  Warning: {backend} backend produced incorrect results!")
                max_diff = np.max(np.abs(result - numpy_result))
                print(f"  Maximum difference: {max_diff}")
            
            print(f"  {backend}: {elapsed:.2f} ms")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for backend in backends:
        # Skip backends with errors
        if any(t is None for t in results[backend]):
            continue
        plt.plot([s[0] for s in sizes], results[backend], marker='o', label=backend)
    
    plt.title("Matrix Multiplication Performance")
    plt.xlabel("Matrix Size (N×N)")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig("matrix_multiplication_benchmark.png")
    print(f"\nBenchmark plot saved to matrix_multiplication_benchmark.png")


def automatic_backend_selection():
    """
    Demonstrate automatic backend selection based on operation and hardware.
    """
    print("\n=== Automatic Backend Selection ===")
    
    # Create test data
    size = 1_000_000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Use automatic backend selection
    print("Performing vector addition with automatic backend selection...")
    start = time.time()
    result = vector_add(a, b)
    end = time.time()
    
    elapsed = (end - start) * 1000  # Convert to milliseconds
    print(f"Execution time: {elapsed:.2f} ms")
    
    # Compare with NumPy
    start = time.time()
    numpy_result = a + b
    end = time.time()
    
    numpy_elapsed = (end - start) * 1000  # Convert to milliseconds
    print(f"NumPy execution time: {numpy_elapsed:.2f} ms")
    
    # Calculate speedup
    speedup = numpy_elapsed / elapsed
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify result
    if np.allclose(result, numpy_result, rtol=1e-5, atol=1e-5):
        print("Results match NumPy calculation")
    else:
        print("Warning: Results differ from NumPy calculation")
        max_diff = np.max(np.abs(result - numpy_result))
        print(f"Maximum difference: {max_diff}")


def hardware_specific_optimizations():
    """
    Demonstrate hardware-specific optimizations based on detected platform.
    """
    print("\n=== Hardware-Specific Optimizations ===")
    
    platform_name, arch = detect_platform()
    print(f"Detected platform: {platform_name} ({arch})")
    
    if arch == "arm64":
        if platform_name == "Apple Silicon":
            print("Using Apple Silicon optimizations...")
            
            # Matrix multiplication optimized for Apple Silicon
            size = 1000
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            
            # Try using Apple-specific context if available
            try:
                backend_name = "m1"  # Default to M1, works on all Apple Silicon
                with prism.use_backend(backend_name):
                    print(f"Using {backend_name} backend for matrix multiplication...")
                    start = time.time()
                    result = matrix_multiply(a, b)
                    end = time.time()
                
                elapsed = (end - start) * 1000  # Convert to milliseconds
                print(f"Execution time with {backend_name} backend: {elapsed:.2f} ms")
                
                # Compare with NumPy
                start = time.time()
                numpy_result = np.matmul(a, b)
                end = time.time()
                
                numpy_elapsed = (end - start) * 1000  # Convert to milliseconds
                print(f"NumPy execution time: {numpy_elapsed:.2f} ms")
                
                # Calculate speedup
                speedup = numpy_elapsed / elapsed
                print(f"Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"Apple Silicon backend not available: {str(e)}")
        
        elif "graviton" in platform_name.lower() or platform_name == "ARM":
            print("Using ARM-specific optimizations...")
            
            # Try using ARM-specific context
            try:
                with prism.use_backend("assembly"):
                    print("Using assembly backend with ARM NEON/SVE optimizations...")
                    
                    # Vector addition with ARM optimizations
                    size = 10_000_000
                    a = np.random.rand(size).astype(np.float32)
                    b = np.random.rand(size).astype(np.float32)
                    
                    start = time.time()
                    result = vector_add(a, b)
                    end = time.time()
                    
                    elapsed = (end - start) * 1000  # Convert to milliseconds
                    print(f"Vector addition execution time: {elapsed:.2f} ms")
                    
                    # Compare with NumPy
                    start = time.time()
                    numpy_result = a + b
                    end = time.time()
                    
                    numpy_elapsed = (end - start) * 1000  # Convert to milliseconds
                    print(f"NumPy execution time: {numpy_elapsed:.2f} ms")
                    
                    # Calculate speedup
                    speedup = numpy_elapsed / elapsed
                    print(f"Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"ARM assembly backend not available: {str(e)}")
    
    elif arch == "x86_64":
        print("Using x86_64-specific optimizations...")
        
        # Try using x86_64-specific optimizations
        try:
            with prism.use_backend("c"):
                print("Using C/C++ backend with x86_64 optimizations (AVX/AVX2/AVX-512)...")
                
                # Matrix multiplication with x86_64 optimizations
                size = 1000
                a = np.random.rand(size, size).astype(np.float32)
                b = np.random.rand(size, size).astype(np.float32)
                
                start = time.time()
                result = matrix_multiply(a, b)
                end = time.time()
                
                elapsed = (end - start) * 1000  # Convert to milliseconds
                print(f"Matrix multiplication execution time: {elapsed:.2f} ms")
                
                # Compare with NumPy
                start = time.time()
                numpy_result = np.matmul(a, b)
                end = time.time()
                
                numpy_elapsed = (end - start) * 1000  # Convert to milliseconds
                print(f"NumPy execution time: {numpy_elapsed:.2f} ms")
                
                # Calculate speedup
                speedup = numpy_elapsed / elapsed
                print(f"Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"C/C++ backend not available: {str(e)}")


def main():
    """Main function to run the example."""
    print("Prism Framework - Unified Backend Example")
    print("========================================")
    
    # List available backends
    list_available_backends()
    
    # Run automatic backend selection
    automatic_backend_selection()
    
    # Demonstrate hardware-specific optimizations
    hardware_specific_optimizations()
    
    # Run benchmarks
    benchmark_vector_addition()
    benchmark_matrix_multiplication()


if __name__ == "__main__":
    main()