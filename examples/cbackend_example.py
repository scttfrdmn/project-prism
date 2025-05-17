"""
Example usage of the C/C++ backend in Prism framework.

This example demonstrates how to:
1. Generate C/C++ code for different architectures
2. Compile and execute the C/C++ code
3. Use the high-level interface for common operations
"""

import os
import sys
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Prism framework components
from src.backends.cbackend import (
    CCodeGenerator,
    CompilerWrapper,
    CBackend,
    get_template
)


def code_generation_example():
    """
    Example demonstrating C/C++ code generation.
    """
    print("\n=== C/C++ Code Generation Example ===")
    
    # Initialize code generator
    generator = CCodeGenerator()
    print(f"Generating code for architecture: {generator.arch}")
    
    # Generate vector addition function
    vector_add_code = generator.generate_vector_add_function("vector_add")
    print("\nGenerated Vector Addition Function:")
    print("----------------------------------")
    print(vector_add_code[:500] + "...\n")  # Show first 500 chars
    
    # Generate matrix multiplication function
    matrix_multiply_code = generator.generate_matrix_multiply_function("matrix_multiply")
    print("\nGenerated Matrix Multiplication Function:")
    print("---------------------------------------")
    print(matrix_multiply_code[:500] + "...\n")  # Show first 500 chars
    
    # Generate complete source code
    full_source = generator.generate_full_source(
        function_name="vector_add",
        operation="vector_add",
        with_main=True
    )
    
    # Save to a file for inspection
    output_file = "vector_add_example.c"
    with open(output_file, "w") as f:
        f.write(full_source)
    
    print(f"Full source code saved to {output_file}")


def compilation_example():
    """
    Example demonstrating C/C++ code compilation.
    """
    print("\n=== C/C++ Compilation Example ===")
    
    try:
        # Initialize compiler
        compiler = CompilerWrapper()
        print(f"Using compiler: {compiler.compiler}")
        print(f"Compiler version: {compiler.get_compiler_version()}")
        
        # Generate code
        generator = CCodeGenerator()
        source_code = generator.generate_full_source(
            function_name="vector_add",
            operation="vector_add"
        )
        
        # Compile the code
        output_file = "vector_add_lib"
        if sys.platform == 'darwin':
            output_file += '.dylib'
        elif sys.platform == 'win32':
            output_file += '.dll'
        else:
            output_file += '.so'
        
        try:
            lib_path = compiler.compile(source_code, output_file=output_file)
            print(f"Compilation successful: {lib_path}")
            
            # Clean up
            if os.path.exists(lib_path):
                os.unlink(lib_path)
                print(f"Removed compiled library: {lib_path}")
            
        except Exception as e:
            print(f"Compilation failed: {str(e)}")
    
    except Exception as e:
        print(f"Failed to initialize compiler: {str(e)}")
        print("Skipping compilation example...")


def vector_add_example():
    """
    Example demonstrating vector addition using the C/C++ backend.
    """
    print("\n=== Vector Addition Example ===")
    
    try:
        # Initialize backend
        backend = CBackend()
        
        # Create test data
        size = 1000000
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)
        
        # Expected result for verification
        expected_result = a + b
        
        print(f"Performing vector addition with {size} elements")
        
        # Time the C/C++ implementation
        start = time.time()
        result = backend.vector_add(a, b)
        end = time.time()
        
        c_time = (end - start) * 1000  # Convert to ms
        print(f"C/C++ implementation time: {c_time:.2f} ms")
        
        # Time the NumPy implementation
        start = time.time()
        numpy_result = a + b
        end = time.time()
        
        numpy_time = (end - start) * 1000  # Convert to ms
        print(f"NumPy implementation time: {numpy_time:.2f} ms")
        
        # Verify the result
        if np.allclose(result, expected_result, rtol=1e-5, atol=1e-5):
            print("✓ Result matches numpy calculation")
        else:
            print("✗ Result differs from numpy calculation")
            max_diff = np.max(np.abs(result - expected_result))
            print(f"  Maximum difference: {max_diff}")
        
        # Calculate speedup
        speedup = numpy_time / c_time
        print(f"Speedup over NumPy: {speedup:.2f}x")
        
        # Clean up
        backend.cleanup()
    
    except Exception as e:
        print(f"Error in vector addition example: {str(e)}")


def matrix_multiply_example():
    """
    Example demonstrating matrix multiplication using the C/C++ backend.
    """
    print("\n=== Matrix Multiplication Example ===")
    
    try:
        # Initialize backend
        backend = CBackend()
        
        # Create test data (small matrices for demonstration)
        m, n, k = 500, 500, 500
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        print(f"Performing matrix multiplication with {m}x{k} and {k}x{n} matrices")
        
        # Time the C/C++ implementation
        start = time.time()
        result = backend.matrix_multiply(a, b)
        end = time.time()
        
        c_time = (end - start) * 1000  # Convert to ms
        print(f"C/C++ implementation time: {c_time:.2f} ms")
        
        # Time the NumPy implementation
        start = time.time()
        numpy_result = np.matmul(a, b)
        end = time.time()
        
        numpy_time = (end - start) * 1000  # Convert to ms
        print(f"NumPy implementation time: {numpy_time:.2f} ms")
        
        # Verify the result
        if np.allclose(result, numpy_result, rtol=1e-4, atol=1e-4):
            print("✓ Result matches numpy calculation")
        else:
            print("✗ Result differs from numpy calculation")
            max_diff = np.max(np.abs(result - numpy_result))
            print(f"  Maximum difference: {max_diff}")
        
        # Calculate speedup
        speedup = numpy_time / c_time
        print(f"Speedup over NumPy: {speedup:.2f}x")
        
        # Clean up
        backend.cleanup()
    
    except Exception as e:
        print(f"Error in matrix multiplication example: {str(e)}")


def benchmark_example():
    """
    Example benchmarking C/C++ backend against NumPy.
    """
    print("\n=== Benchmark Example ===")
    
    try:
        # Initialize backend
        backend = CBackend()
        
        # Vector sizes to test
        sizes = [10000, 100000, 1000000, 10000000]
        
        # Results
        c_times = []
        numpy_times = []
        
        for size in sizes:
            print(f"\nTesting vector addition with size {size}")
            
            # Create test data
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)
            
            # Time the C/C++ implementation
            start = time.time()
            result = backend.vector_add(a, b)
            end = time.time()
            
            c_time = (end - start) * 1000  # Convert to ms
            c_times.append(c_time)
            print(f"C/C++ implementation time: {c_time:.2f} ms")
            
            # Time the NumPy implementation
            start = time.time()
            numpy_result = a + b
            end = time.time()
            
            numpy_time = (end - start) * 1000  # Convert to ms
            numpy_times.append(numpy_time)
            print(f"NumPy implementation time: {numpy_time:.2f} ms")
            
            # Calculate speedup
            speedup = numpy_time / c_time
            print(f"Speedup over NumPy: {speedup:.2f}x")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, c_times, marker='o', label='C/C++ Backend')
        plt.plot(sizes, numpy_times, marker='x', label='NumPy')
        plt.title("Vector Addition Performance Comparison")
        plt.xlabel("Vector Size")
        plt.ylabel("Execution Time (ms)")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        
        # Save the plot
        plt.savefig("cbackend_benchmark.png")
        print("\nBenchmark plot saved to cbackend_benchmark.png")
        
        # Clean up
        backend.cleanup()
    
    except Exception as e:
        print(f"Error in benchmark example: {str(e)}")


def template_example():
    """
    Example demonstrating the use of code templates.
    """
    print("\n=== Template Example ===")
    
    # Get a template for vector addition
    vector_add_template = get_template("vector_add")
    print("\nVector Addition Template:")
    print("------------------------")
    print(vector_add_template[:500] + "...\n")  # Show first 500 chars
    
    # Get a template for matrix multiplication
    matrix_multiply_template = get_template("matrix_multiply")
    print("\nMatrix Multiplication Template:")
    print("-----------------------------")
    print(matrix_multiply_template[:500] + "...\n")  # Show first 500 chars


def main():
    """Main function."""
    print("C/C++ Backend Examples")
    print("======================")
    
    # Run examples
    code_generation_example()
    compilation_example()
    vector_add_example()
    matrix_multiply_example()
    benchmark_example()
    template_example()


if __name__ == "__main__":
    main()