"""
Example usage of the assembly backend in Prism framework.

This example demonstrates how to:
1. Generate assembly code for different architectures
2. Assemble and execute the assembly code using the runtime
3. Use pre-defined assembly code templates for common operations
"""

import os
import sys
import ctypes
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Prism framework components
from src.backends.asmbackend import (
    ARMAssemblyGenerator, 
    X86AssemblyGenerator,
    ARMExamples,
    X86Examples, 
    get_examples,
    get_assembler,
    AssemblyBackend
)


def vector_add_example():
    """
    Example demonstrating vector addition using the assembly backend.
    """
    print("\n=== Vector Addition Example ===")
    
    # Create test data
    size = 1024
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    c = np.zeros(size, dtype=np.float32)
    
    # Expected result for verification
    expected_result = a + b
    
    # Initialize the backend based on architecture
    backend = AssemblyBackend()
    print(f"Using backend for architecture: {backend.arch}")
    
    # Get examples for the current architecture
    examples = get_examples()
    
    # Generate assembly code for vector addition
    if backend.arch == "arm64":
        # For ARM architecture
        asm_code = ARMExamples.vector_add_float32("a", "b", "c", size)
    else:
        # For x86_64 architecture
        asm_code = X86Examples.vector_add_float32("a", "b", "c", size, 
                                                 syntax=backend.syntax)
    
    print(f"Generated assembly code for vector addition")
    
    # Compile and load the assembly code
    kernel = backend.compile_and_load(
        asm_code=asm_code,
        function_name="vector_add_float32",
        arg_types=[
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ],
        return_type=None
    )
    
    print("Compiled assembly code to shared library")
    
    # Create ctypes pointers to the numpy arrays
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the assembly function
    kernel(a_ptr, b_ptr, c_ptr, size)
    
    print("Executed assembly function")
    
    # Verify the result
    if np.allclose(c, expected_result, rtol=1e-5, atol=1e-5):
        print("✓ Result matches numpy calculation")
    else:
        print("✗ Result differs from numpy calculation")
        max_diff = np.max(np.abs(c - expected_result))
        print(f"  Maximum difference: {max_diff}")
    
    # Cleanup
    kernel.cleanup()
    print("Cleaned up resources")


def matrix_multiply_example():
    """
    Example demonstrating matrix multiplication using the assembly backend.
    """
    print("\n=== Matrix Multiplication Example ===")
    
    # Create test data (small matrices for demonstration)
    m, n, k = 16, 16, 16
    a = np.random.rand(m, k).astype(np.float32)
    b = np.random.rand(k, n).astype(np.float32)
    c = np.zeros((m, n), dtype=np.float32)
    
    # Expected result for verification
    expected_result = np.matmul(a, b)
    
    # Initialize the backend based on architecture
    backend = AssemblyBackend()
    
    # Generate assembly code for matrix multiplication
    if backend.arch == "arm64":
        # For ARM architecture
        asm_code = ARMExamples.matrix_multiply_float32("a", "b", "c", m, n, k)
    else:
        # For x86_64 architecture
        asm_code = X86Examples.matrix_multiply_float32("a", "b", "c", m, n, k, 
                                                     syntax=backend.syntax)
    
    print(f"Generated assembly code for matrix multiplication")
    
    # Compile and load the assembly code
    kernel = backend.compile_and_load(
        asm_code=asm_code,
        function_name="matrix_multiply_float32",
        arg_types=[
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int
        ],
        return_type=None
    )
    
    print("Compiled assembly code to shared library")
    
    # Create ctypes pointers to the numpy arrays (flattened)
    a_ptr = a.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call the assembly function
    kernel(a_ptr, b_ptr, c_ptr, m, n, k)
    
    print("Executed assembly function")
    
    # Verify the result
    if np.allclose(c, expected_result, rtol=1e-4, atol=1e-4):
        print("✓ Result matches numpy calculation")
    else:
        print("✗ Result differs from numpy calculation")
        max_diff = np.max(np.abs(c - expected_result))
        print(f"  Maximum difference: {max_diff}")
    
    # Cleanup
    kernel.cleanup()
    print("Cleaned up resources")


def direct_backend_example():
    """
    Example demonstrating direct use of the assembly backend with
    a simple function.
    """
    print("\n=== Direct Backend Example ===")
    
    # Initialize the backend based on architecture
    backend = AssemblyBackend()
    
    # Define a simple function that squares a number
    if backend.arch == "arm64":
        # ARM assembly for squaring a float
        asm_code = """
        .global square_float
        .type square_float, %function
        
        square_float:
            fmul s0, s0, s0
            ret
        """
    else:
        # x86_64 assembly for squaring a float
        if backend.syntax == "intel":
            asm_code = """
            .intel_syntax noprefix
            .global square_float
            .type square_float, @function
            
            square_float:
                mulss xmm0, xmm0
                ret
            """
        else:
            asm_code = """
            .att_syntax
            .global square_float
            .type square_float, @function
            
            square_float:
                mulss %xmm0, %xmm0
                ret
            """
    
    print("Generated assembly code for square function")
    
    # Execute the assembly code directly with the backend
    result = backend.execute(
        asm_code=asm_code,
        function_name="square_float",
        args=[ctypes.c_float(4.0)],
        arg_types=[ctypes.c_float],
        return_type=ctypes.c_float
    )
    
    print(f"Executed assembly function: 4.0² = {result}")
    print(f"Expected result: {4.0 * 4.0}")
    
    # Cleanup backend resources
    backend.cleanup()
    print("Cleaned up resources")


if __name__ == "__main__":
    # Run the examples
    vector_add_example()
    matrix_multiply_example()
    direct_backend_example()