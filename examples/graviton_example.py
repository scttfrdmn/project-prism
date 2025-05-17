#!/usr/bin/env python3
"""
Example usage of AWS Graviton devices with Prism framework.

This script demonstrates how to work with different Graviton versions
and use the device context manager to switch between them.
"""

import numpy as np
import time

import src as prism
from src.backends.graviton import (
    GravitonDevice,
    Graviton3Device,
    Graviton3EDevice,
    Graviton4Device
)

def print_device_info(device, label=None):
    """Print information about a device."""
    if label:
        print(f"\n=== {label} ===")
    
    capabilities = device.get_capabilities()
    print(f"Device: {capabilities['name']}")
    print(f"Architecture: {capabilities['architecture']}")
    print(f"Cores: {capabilities['compute_units']}")
    print(f"Vector Extensions: {capabilities['vector_extension']}")
    print(f"Vector Width: {capabilities['vector_width_bits']} bits")
    
    if 'sve_supported' in capabilities and capabilities['sve_supported']:
        print(f"SVE Support: Yes (vector length: {capabilities['sve_vector_length']} bits)")
        if 'sve2_supported' in capabilities and capabilities['sve2_supported']:
            print("SVE2 Support: Yes")
    
    if 'bf16_support' in capabilities and capabilities['bf16_support']:
        print("BF16 Support: Yes")
    
    if 'fp8_support' in capabilities and capabilities['fp8_support']:
        print("FP8 Support: Yes")
    
    if 'vector_performance_boost' in capabilities:
        print(f"Vector Performance Boost: {capabilities['vector_performance_boost']}")
    
    if 'hpc_optimized' in capabilities and capabilities['hpc_optimized']:
        print("HPC Optimized: Yes")
    
    if 'enhanced_networking' in capabilities and capabilities['enhanced_networking']:
        print(f"Enhanced Networking: Yes (Bandwidth: {capabilities['network_bandwidth_gbps']} Gbps)")
    
    print(f"Optimal Memory Alignment: {device.get_optimal_memory_alignment()} bytes")
    print()

def vector_addition_benchmark(device, size=10000000, iterations=5):
    """
    Run a simple vector addition benchmark on the specified device.
    
    Args:
        device: Device to run the benchmark on
        size: Size of vectors
        iterations: Number of iterations
        
    Returns:
        Average time per iteration in milliseconds
    """
    print(f"Running vector addition benchmark on {device}...")
    print(f"Vector size: {size:,} elements")
    
    # Create input data
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    c = np.zeros_like(a)
    
    # Convert to device memory (no-op for Graviton, but consistent API)
    a_dev = device.allocate_memory(a.nbytes)
    b_dev = device.allocate_memory(b.nbytes)
    c_dev = device.allocate_memory(c.nbytes)
    
    device.copy_to_device(a, a_dev, a.nbytes)
    device.copy_to_device(b, b_dev, b.nbytes)
    
    # Warmup iteration
    np.add(a_dev, b_dev, out=c_dev)
    
    # Benchmark iterations
    times = []
    for i in range(iterations):
        start_time = time.time()
        np.add(a_dev, b_dev, out=c_dev)
        device.synchronize()  # No-op for Graviton, but consistent API
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Elements per second: {size / (avg_time / 1000) / 1e6:.2f} million")
    print()
    
    return avg_time

def main():
    """Main function to demonstrate Graviton device usage."""
    print("=== AWS Graviton Example with Prism Framework ===\n")
    
    # Get default device
    default_device = prism.get_active_device()
    print_device_info(default_device, "Default Device")
    
    # Use context manager to switch to Graviton 3
    with prism.graviton3() as g3:
        print_device_info(g3, "Graviton 3")
        vector_addition_benchmark(g3)
        
        # Get SVE configuration
        sve_config = g3.get_sve_configuration()
        print(f"SVE Configuration: {sve_config}\n")
    
    # Use context manager to switch to Graviton 3E
    with prism.graviton3e() as g3e:
        print_device_info(g3e, "Graviton 3E")
        vector_addition_benchmark(g3e)
        
        # Use HPC optimization
        print("Applying HPC optimization for MPI workloads...")
        g3e.optimize_for_hpc(hpc_type="mpi")
        
        # Run benchmark again with HPC optimization
        vector_addition_benchmark(g3e)
    
    # Use context manager to switch to Graviton 4
    with prism.graviton4() as g4:
        print_device_info(g4, "Graviton 4")
        vector_addition_benchmark(g4)
        
        # Enable confidential computing
        print("Enabling confidential computing...")
        g4.enable_confidential_computing()
    
    # Compare all Graviton versions
    print("=== Comparing All Graviton Versions ===\n")
    
    results = {}
    
    # Loop through all Graviton versions
    for version in [1, 2, 3, "3E", 4]:
        with prism.graviton(version=version) as device:
            name = f"Graviton {version}"
            print_device_info(device, name)
            
            # Run benchmark
            avg_time = vector_addition_benchmark(device)
            results[name] = avg_time
    
    # Print comparison
    print("=== Performance Comparison ===\n")
    print("Version         | Average Time (ms) | Relative Performance")
    print("-" * 65)
    
    # Find best performance for normalization
    best_time = min(results.values())
    
    for name, time in results.items():
        relative = best_time / time
        print(f"{name:15} | {time:16.2f} | {relative:20.2f}x")
    
    print("\nNote: Relative performance is normalized to the fastest device.")
    print("      Higher values indicate better performance.")

if __name__ == "__main__":
    main()