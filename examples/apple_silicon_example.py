#!/usr/bin/env python3
"""
Example usage of Apple Silicon devices with Prism framework.

This script demonstrates how to work with different Apple Silicon versions
and use the device context manager to switch between them.
"""

import numpy as np
import time
import platform
import os

import src as prism
from src.backends.apple_silicon import (
    AppleSiliconDevice,
    M1Device, M1ProDevice, M1MaxDevice, M1UltraDevice,
    M2Device, M2ProDevice, M2MaxDevice, M2UltraDevice,
    M3Device, M3ProDevice, M3MaxDevice, M3UltraDevice,
    M4Device, M4ProDevice, M4MaxDevice
)


def print_device_info(device, label=None):
    """Print information about a device."""
    if label:
        print(f"\n=== {label} ===")
    
    capabilities = device.get_capabilities()
    print(f"Device: {capabilities['name']}")
    print(f"Architecture: {capabilities['architecture']}")
    print(f"Compute Units: {capabilities['compute_units']} cores (P-cores: {device.features.get('performance_cores', 0)}, E-cores: {device.features.get('efficiency_cores', 0)})")
    print(f"Vector Extensions: {capabilities['vector_extension']}")
    print(f"Vector Width: {capabilities['vector_width_bits']} bits")
    
    # Print Neural Engine information
    if 'neural_engine_tops' in capabilities:
        print(f"Neural Engine: {capabilities['neural_engine_tops']} TOPS")
    
    # Print GPU information
    if 'gpu_cores' in capabilities:
        print(f"GPU Cores: {capabilities['gpu_cores']}")
    
    # Print ray tracing and mesh shading if supported
    if capabilities.get('ray_tracing', False):
        print("Hardware Ray Tracing: Supported")
    
    if capabilities.get('mesh_shading', False):
        print("Mesh Shading: Supported")
    
    if capabilities.get('dynamic_caching', False):
        print("Dynamic Caching: Supported")
    
    # Print memory information
    memory_info = device.get_memory_info()
    if 'total_memory_gb' in memory_info:
        print(f"Memory: {memory_info['total_memory_gb']:.1f} GB")
    
    if 'memory_bandwidth_gbps' in memory_info:
        print(f"Memory Bandwidth: {memory_info['memory_bandwidth_gbps']} GB/s")
    
    # Print media engines if available
    if 'media_engines' in capabilities:
        print(f"Media Engines: {capabilities['media_engines']}")
    
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
    
    # Convert to device memory (no-op for Apple Silicon, but consistent API)
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
        device.synchronize()  # No-op for Apple Silicon, but consistent API
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Elements per second: {size / (avg_time / 1000) / 1e6:.2f} million")
    print()
    
    return avg_time


def neural_engine_benchmark(device, size=1000, iterations=5):
    """
    Simulate a Neural Engine benchmark on the specified device.
    
    Note: This is a simulated benchmark. In a real implementation, we would use
    CoreML to run a model on the Neural Engine, but this requires more setup.
    
    Args:
        device: Device to run the benchmark on
        size: Size of matrix
        iterations: Number of iterations
        
    Returns:
        Average time per iteration in milliseconds
    """
    neural_tops = device.get_capabilities().get("neural_engine_tops", 0)
    
    if neural_tops == 0:
        print("Neural Engine not available on this device")
        return 0
    
    print(f"Running Neural Engine benchmark on {device}...")
    print(f"Neural Engine performance: {neural_tops} TOPS")
    print(f"Matrix size: {size}x{size}")
    
    # Create random matrices
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    # Simulate matrix multiplication with scaling factor based on Neural Engine TOPS
    # This is just a simulation - real performance would be measured using CoreML
    scaling_factor = neural_tops / 11.0  # Normalize to M1's 11 TOPS
    
    # Benchmark iterations
    times = []
    for i in range(iterations):
        start_time = time.time()
        # Simulate matrix multiplication
        c = np.matmul(a, b)
        # Apply artificial speedup based on Neural Engine capabilities
        # This is just for simulation purposes
        time.sleep(max(0.001 / scaling_factor, 0.0001))
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms (simulated)")
    print()
    
    return avg_time


def main():
    """Main function to demonstrate Apple Silicon device usage."""
    print("=== Apple Silicon Example with Prism Framework ===\n")
    
    # Check if running on macOS first
    if platform.system() != "Darwin":
        print("This example is designed to run on macOS with Apple Silicon processors.")
        print("Since you're not running on macOS, we'll use environment variables to simulate Apple Silicon detection.\n")
        os.environ["APPLE_SILICON_VERSION"] = "M1"
    
    # Get default device
    default_device = prism.get_active_device()
    print_device_info(default_device, "Default Device")
    
    # Explicitly create AppleSiliconDevice instance
    apple_device = AppleSiliconDevice()
    print_device_info(apple_device, "Detected Apple Silicon Device")
    
    print("\n=== Using Context Managers for Different Apple Silicon Versions ===\n")
    print("Note: Context managers allow for temporary device switching")
    
    # Use context manager to switch to M1
    # We might not be running on actual Apple Silicon, so we'll catch exceptions
    try:
        with prism.m1() as device:
            print_device_info(device, "Apple M1")
            vector_addition_benchmark(device, size=1000000)
    except Exception as e:
        print(f"Error using M1 context: {e}")
    
    # Use context manager to switch to M2
    try:
        with prism.m2() as device:
            print_device_info(device, "Apple M2")
            vector_addition_benchmark(device, size=1000000)
    except Exception as e:
        print(f"Error using M2 context: {e}")
    
    # Use context manager to switch to M3
    try:
        with prism.m3() as device:
            print_device_info(device, "Apple M3")
            
            # Check for ray tracing support
            if device.supports_ray_tracing():
                print("Demonstrating ray tracing workload...")
                # Simulated ray tracing workload
                print("Ray tracing hardware acceleration active")
            
            vector_addition_benchmark(device, size=1000000)
    except Exception as e:
        print(f"Error using M3 context: {e}")
    
    # Use context manager to switch to M4
    try:
        with prism.m4() as device:
            print_device_info(device, "Apple M4")
            
            # Run Neural Engine benchmark
            neural_engine_benchmark(device)
            
            vector_addition_benchmark(device, size=1000000)
    except Exception as e:
        print(f"Error using M4 context: {e}")
    
    # Compare Pro, Max, and Ultra variants of a specific generation
    print("\n=== Comparing M3 Series Variants ===\n")
    
    try:
        variants = [
            ("M3", prism.m3),
            ("M3 Pro", prism.m3_pro),
            ("M3 Max", prism.m3_max),
            ("M3 Ultra", prism.m3_ultra),
        ]
        
        for name, context_manager in variants:
            try:
                with context_manager() as device:
                    print_device_info(device, name)
            except Exception as e:
                print(f"Error using {name} context: {e}")
    except Exception as e:
        print(f"Error comparing M3 variants: {e}")
    
    # Using automatic backend selection
    print("\n=== Automatic Backend Selection ===\n")
    
    # For inference workloads
    try:
        inference_device = prism.create_optimal_device(
            workload_type="inference",
            batch_size=32,
            precision="fp16"
        )
        print_device_info(inference_device, "Optimal Device for Inference")
    except Exception as e:
        print(f"Error creating optimal device for inference: {e}")
    
    # For ray tracing workloads
    try:
        ray_tracing_device = prism.create_optimal_device(
            workload_type="custom_logic",
            ray_tracing=True
        )
        print_device_info(ray_tracing_device, "Optimal Device for Ray Tracing")
    except Exception as e:
        print(f"Error creating optimal device for ray tracing: {e}")


if __name__ == "__main__":
    main()