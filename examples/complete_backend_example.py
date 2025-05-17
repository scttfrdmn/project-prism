"""
Complete example demonstrating the integrated low-level backends in Prism.

This example shows how to:
1. Use the low-level backends via direct function calls
2. Switch backends using context managers
3. Utilize device-specific optimizations
4. Create custom operations using the assembly backend
"""

import os
import sys
import time
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Prism framework components
import src as prism
from src.core.lowlevel import vector_add, matrix_multiply
from src.backends.asmbackend import AssemblyBackend


def basic_usage_example():
    """
    Demonstrates basic usage of the low-level operations.
    """
    print("\n=== Basic Usage Example ===")
    
    # Create test data
    size = 10000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Use the vector_add function
    print("Vector addition with auto-selected backend:")
    start = time.time()
    result = vector_add(a, b)
    end = time.time()
    
    print(f"Time: {(end - start) * 1000:.2f} ms")
    print(f"Result matches numpy: {np.allclose(result, a + b, rtol=1e-5, atol=1e-5)}")


def backend_switching_example():
    """
    Demonstrates switching between backends using context managers.
    """
    print("\n=== Backend Switching Example ===")
    
    # Create test data
    size = 10000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Use explicit assembly backend
    print("Vector addition with assembly backend:")
    with prism.use_assembly_backend():
        start = time.time()
        result_asm = vector_add(a, b)
        end = time.time()
    
    print(f"Time with assembly backend: {(end - start) * 1000:.2f} ms")
    
    # Compare with numpy
    print("Vector addition with numpy:")
    start = time.time()
    result_numpy = a + b
    end = time.time()
    
    print(f"Time with numpy: {(end - start) * 1000:.2f} ms")
    print(f"Results match: {np.allclose(result_asm, result_numpy, rtol=1e-5, atol=1e-5)}")


def device_specific_example():
    """
    Demonstrates device-specific optimizations for Apple Silicon.
    """
    print("\n=== Device-Specific Example ===")
    
    # Check if running on Apple Silicon
    import platform
    if platform.machine() not in ('arm64', 'aarch64'):
        print("Not running on ARM architecture, skipping device-specific example")
        return
    
    # Create test data
    size = 100
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    
    # Use with Apple Silicon context manager
    print("Matrix multiplication with Apple Silicon optimization:")
    with prism.m1():  # Will use M1 if available, or fallback gracefully
        start = time.time()
        result_m1 = matrix_multiply(a, b)
        end = time.time()
    
    print(f"Time with Apple Silicon context: {(end - start) * 1000:.2f} ms")
    
    # Compare with numpy
    print("Matrix multiplication with numpy:")
    start = time.time()
    result_numpy = np.matmul(a, b)
    end = time.time()
    
    print(f"Time with numpy: {(end - start) * 1000:.2f} ms")
    print(f"Results match: {np.allclose(result_m1, result_numpy, rtol=1e-4, atol=1e-4)}")


def custom_assembly_operation():
    """
    Demonstrates creating a custom operation using the assembly backend directly.
    """
    print("\n=== Custom Assembly Operation Example ===")
    
    # Initialize the assembly backend
    backend = AssemblyBackend()
    
    # Define a simple function to compute the dot product of two vectors
    if backend.arch == "arm64":
        # ARM assembly for dot product
        asm_code = """
        .global dot_product
        .type dot_product, %function
        
        // Function: float dot_product(float* a, float* b, int size)
        // Arguments:
        //   x0 = pointer to vector a
        //   x1 = pointer to vector b
        //   w2 = size of vectors
        
        dot_product:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            
            // Initialize sum to 0
            fmov s0, wzr
            
            // Check if size <= 0
            cmp w2, #0
            ble .done
            
            // Initialize loop counter
            mov w3, #0
            
        .loop:
            // Load values from vectors
            ldr s1, [x0, w3, UXTW #2]
            ldr s2, [x1, w3, UXTW #2]
            
            // Multiply and accumulate
            fmadd s0, s1, s2, s0
            
            // Increment counter and loop
            add w3, w3, #1
            cmp w3, w2
            blt .loop
            
        .done:
            // Return result in s0
            ldp x29, x30, [sp], #16
            ret
        """
    else:
        # x86_64 assembly for dot product
        if backend.syntax == "intel":
            asm_code = """
            .intel_syntax noprefix
            .global dot_product
            .type dot_product, @function
            
            // Function: float dot_product(float* a, float* b, int size)
            // Arguments:
            //   rdi = pointer to vector a
            //   rsi = pointer to vector b
            //   edx = size of vectors
            
            dot_product:
                // Initialize sum to 0
                xorps xmm0, xmm0
                
                // Check if size <= 0
                test edx, edx
                jle done
                
                // Initialize loop counter
                xor eax, eax
                
            loop:
                // Load values from vectors
                movss xmm1, DWORD PTR [rdi + rax*4]
                movss xmm2, DWORD PTR [rsi + rax*4]
                
                // Multiply and accumulate
                mulss xmm1, xmm2
                addss xmm0, xmm1
                
                // Increment counter and loop
                inc eax
                cmp eax, edx
                jl loop
                
            done:
                // Result already in xmm0
                ret
            """
        else:
            asm_code = """
            .att_syntax
            .global dot_product
            .type dot_product, @function
            
            // Function: float dot_product(float* a, float* b, int size)
            // Arguments:
            //   %rdi = pointer to vector a
            //   %rsi = pointer to vector b
            //   %edx = size of vectors
            
            dot_product:
                // Initialize sum to 0
                xorps %xmm0, %xmm0
                
                // Check if size <= 0
                testl %edx, %edx
                jle done
                
                // Initialize loop counter
                xorl %eax, %eax
                
            loop:
                // Load values from vectors
                movss (%rdi,%rax,4), %xmm1
                movss (%rsi,%rax,4), %xmm2
                
                // Multiply and accumulate
                mulss %xmm2, %xmm1
                addss %xmm1, %xmm0
                
                // Increment counter and loop
                incl %eax
                cmpl %edx, %eax
                jl loop
                
            done:
                // Result already in %xmm0
                ret
            """
    
    print(f"Generated assembly code for dot product on {backend.arch}")
    
    # Create test data
    size = 1000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)
    
    # Expected result for verification
    expected = np.dot(a, b)
    
    # Execute the assembly code
    import ctypes
    
    try:
        # Create ctypes pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Time the assembly implementation
        start = time.time()
        result = backend.execute(
            asm_code=asm_code,
            function_name="dot_product",
            args=[a_ptr, b_ptr, ctypes.c_int(size)],
            arg_types=[
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ],
            return_type=ctypes.c_float
        )
        end = time.time()
        
        print(f"Dot product result: {result}")
        print(f"Expected result: {expected}")
        print(f"Results match: {abs(result - expected) < 1e-5}")
        print(f"Time with assembly implementation: {(end - start) * 1000:.2f} ms")
        
        # Compare with numpy
        start = time.time()
        np_result = np.dot(a, b)
        end = time.time()
        
        print(f"Time with numpy: {(end - start) * 1000:.2f} ms")
        
    finally:
        # Clean up resources
        backend.cleanup()


def combined_example():
    """
    Combined example showing all features together.
    """
    print("\n=== Combined Example ===")
    
    # Create test data
    size = 500
    a = np.random.rand(size, size).astype(np.float32)
    b = np.random.rand(size, size).astype(np.float32)
    c = np.random.rand(size).astype(np.float32)
    d = np.random.rand(size).astype(np.float32)
    
    # Use with device and backend context managers
    print("Using combination of device and backend context managers:")
    
    # Try to use M1 with assembly backend
    with prism.m1():
        with prism.use_assembly_backend():
            print("Matrix multiplication:")
            start = time.time()
            result1 = matrix_multiply(a, b)
            end = time.time()
            print(f"  Time: {(end - start) * 1000:.2f} ms")
            
            print("Vector addition:")
            start = time.time()
            result2 = vector_add(c, d)
            end = time.time()
            print(f"  Time: {(end - start) * 1000:.2f} ms")
    
    # Compare with numpy
    print("\nUsing numpy:")
    
    print("Matrix multiplication:")
    start = time.time()
    numpy_result1 = np.matmul(a, b)
    end = time.time()
    print(f"  Time: {(end - start) * 1000:.2f} ms")
    print(f"  Results match: {np.allclose(result1, numpy_result1, rtol=1e-4, atol=1e-4)}")
    
    print("Vector addition:")
    start = time.time()
    numpy_result2 = c + d
    end = time.time()
    print(f"  Time: {(end - start) * 1000:.2f} ms")
    print(f"  Results match: {np.allclose(result2, numpy_result2, rtol=1e-5, atol=1e-5)}")


def main():
    """Main function to run all examples."""
    print("Prism Low-Level Backend Integration Examples")
    print("===========================================")
    
    # Run examples
    basic_usage_example()
    backend_switching_example()
    device_specific_example()
    custom_assembly_operation()
    combined_example()


if __name__ == "__main__":
    main()