# Assembly Language Backend Guide

This guide provides an overview of the Assembly backend in the Prism framework, which allows for generating, assembling, and executing optimized architecture-specific assembly code.

## Overview

The Assembly backend enables Prism to generate highly optimized code for specific processor architectures by directly using assembly language instructions. This approach provides the maximum level of control over hardware features and can achieve performance beyond what's possible with high-level languages.

The backend supports two major architectures:
- **ARM** (ARMv8, ARMv9): Including NEON and SVE vector instructions
- **x86_64**: Including SSE, AVX, AVX2, and AVX-512 vector instructions

## Components

The Assembly backend consists of these key components:

1. **Assembly Code Generators**
   - `ARMAssemblyGenerator`: Generates ARM assembly code
   - `X86AssemblyGenerator`: Generates x86_64 assembly code

2. **Assemblers**
   - `ARMAssembler`: Interfaces with system assemblers for ARM code
   - `X86Assembler`: Interfaces with system assemblers for x86_64 code

3. **Runtime**
   - `AssemblyKernel`: Handles loading and executing compiled assembly code
   - `AssemblyBackend`: High-level interface for the entire process

4. **Examples**
   - Pre-built templates for common operations like vector addition and matrix multiplication

## Usage Examples

### Basic Vector Addition

```python
import numpy as np
import ctypes
from src.backends.asmbackend import AssemblyBackend, get_examples

# Initialize backend and get examples for current architecture
backend = AssemblyBackend()
examples = get_examples()

# Create test data
size = 1024
a = np.random.rand(size).astype(np.float32)
b = np.random.rand(size).astype(np.float32)
c = np.zeros(size, dtype=np.float32)

# Generate appropriate assembly code based on architecture
if backend.arch == "arm64":
    asm_code = examples.vector_add_float32("a", "b", "c", size)
else:
    asm_code = examples.vector_add_float32("a", "b", "c", size, syntax=backend.syntax)

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

# Create ctypes pointers to the numpy arrays
a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Execute the assembly function
kernel(a_ptr, b_ptr, c_ptr, size)

# Clean up resources when done
kernel.cleanup()
```

### Direct Execution

You can also directly execute assembly code with a single call:

```python
import ctypes
from src.backends.asmbackend import AssemblyBackend

backend = AssemblyBackend()

# x86_64 assembly for squaring a float
asm_code = """
.att_syntax
.global square_float
.type square_float, @function

square_float:
    mulss %xmm0, %xmm0
    ret
"""

# Execute the assembly code
result = backend.execute(
    asm_code=asm_code,
    function_name="square_float",
    args=[ctypes.c_float(4.0)],
    arg_types=[ctypes.c_float],
    return_type=ctypes.c_float
)

print(f"4.0² = {result}")  # Output: 4.0² = 16.0
```

## Architecture-Specific Optimizations

### ARM Assembly Optimizations

ARM assembly code can leverage:

- **NEON SIMD Instructions**: 128-bit SIMD operations (available on all ARMv8+)
- **SVE/SVE2 Vector Instructions**: Scalable vector extensions (available on newer ARMv8.2+ and ARMv9)
- **Hardware-specific Features**: Like SVE2 for matrix operations on Graviton 3/3E/4

Example of NEON vector addition:
```
// NEON Vector Addition
ld1 {v0.4s}, [x0], #16
ld1 {v1.4s}, [x1], #16
fadd v2.4s, v0.4s, v1.4s
st1 {v2.4s}, [x2], #16
```

### x86_64 Assembly Optimizations

x86_64 assembly code can leverage:

- **SSE Instructions**: 128-bit SIMD operations
- **AVX/AVX2 Instructions**: 256-bit SIMD operations
- **AVX-512 Instructions**: 512-bit SIMD operations (available on newer Intel CPUs)

Example of AVX vector addition:
```
// AVX Vector Addition
vmovups (%rdi), %ymm0
vmovups (%rsi), %ymm1
vaddps %ymm1, %ymm0, %ymm2
vmovups %ymm2, (%rdx)
```

## Requirements

To use the Assembly backend, the following system requirements must be met:

1. **Assembler**: GNU Assembler (as) or LLVM Assembler (llvm-as)
2. **Compiler/Linker**: GCC/Clang for Unix/Linux/macOS or MSVC for Windows
3. **Python Libraries**: NumPy and ctypes (part of standard library)

## Best Practices

1. **Use Pre-defined Templates**: When possible, use the provided example templates which are already optimized for different architectures.

2. **Instruction Set Detection**: Let the backend automatically detect available instruction sets to ensure compatibility.

3. **Resource Management**: Always call `cleanup()` on kernels and the backend when done to properly release resources.

4. **Error Handling**: Wrap assembly code execution in try/except blocks to handle potential assembler or runtime errors.

5. **Benchmarking**: Compare the performance of assembly implementations against high-level implementations to ensure the optimization is worthwhile.

## Advanced Topics

### Custom Assembly Operations

To implement a custom operation:

1. Create a generator function that produces the appropriate assembly code for your operation.
2. Use `AssemblyBackend.compile_and_load()` to compile and load the code.
3. Define the correct argument and return types for the function.
4. Execute the kernel with the appropriate data.

### Cross-Platform Considerations

When writing cross-platform assembly code:

1. Check the current architecture before generating code.
2. For x86_64, decide on the appropriate syntax (AT&T or Intel).
3. Use the architecture-specific generator for optimal code.
4. Consider platform-specific calling conventions.

## Troubleshooting

Common issues and solutions:

1. **Assembler Not Found**: Ensure GNU Assembler (as) or LLVM Assembler (llvm-as) is installed and in your PATH.
2. **Linker Errors**: Verify that you have the appropriate compiler/linker for your platform.
3. **Segmentation Faults**: Check that your assembly code properly follows the calling convention for your architecture.
4. **Incorrect Results**: Ensure that memory alignment is correct and that vector registers are being used properly.

## Further Reading

- [ARM Architecture Reference Manual](https://developer.arm.com/documentation/ddi0487/latest)
- [Intel Architecture Software Developer's Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
- [GNU Assembler Documentation](https://sourceware.org/binutils/docs/as/)
- [LLVM Assembler Documentation](https://llvm.org/docs/CommandGuide/llvm-as.html)