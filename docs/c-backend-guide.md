# C/C++ Backend Guide

This guide provides an overview of the C/C++ backend in the Prism framework, which allows for generating, compiling, and executing optimized architecture-specific C/C++ code.

## Overview

The C/C++ backend enables Prism to generate highly optimized code using architecture-specific SIMD instructions and compiler optimizations. This approach provides excellent performance while maintaining better portability than direct assembly code.

The backend supports:
- Architecture-specific SIMD optimizations (NEON, SVE, SSE, AVX, AVX-512)
- Dynamic code generation based on detected hardware capabilities
- Cross-platform compilation and execution
- Integration with the Prism operation framework

## Components

The C/C++ backend consists of four main components:

1. **Code Generator** (`CCodeGenerator`): Generates architecture-specific C/C++ code with appropriate SIMD instructions.

2. **Compiler Wrapper** (`CompilerWrapper`): Interfaces with system compilers (GCC, Clang, MSVC) and manages compilation with appropriate optimization flags.

3. **Runtime System** (`CompiledKernel`, `CBackend`): Loads and executes compiled C/C++ code using Python's ctypes library.

4. **Templates** (`templates`): Pre-built code templates for common operations.

## Usage Examples

### Basic Vector Addition

```python
import numpy as np
from src.backends.cbackend import CBackend

# Initialize backend
backend = CBackend()

# Create test data
a = np.random.rand(1000000).astype(np.float32)
b = np.random.rand(1000000).astype(np.float32)

# Perform vector addition
result = backend.vector_add(a, b)

# Clean up when done
backend.cleanup()
```

### Matrix Multiplication

```python
import numpy as np
from src.backends.cbackend import CBackend

# Initialize backend
backend = CBackend()

# Create test data
a = np.random.rand(1000, 1000).astype(np.float32)
b = np.random.rand(1000, 1000).astype(np.float32)

# Perform matrix multiplication
result = backend.matrix_multiply(a, b)

# Clean up when done
backend.cleanup()
```

### Custom Code Generation and Compilation

```python
import ctypes
import numpy as np
from src.backends.cbackend import CCodeGenerator, CompilerWrapper, CompiledKernel

# Generate custom C code
generator = CCodeGenerator()
source_code = generator.generate_full_source(
    function_name="custom_function",
    operation="vector_add",
    data_type="float"
)

# Compile the code
compiler = CompilerWrapper()
lib_path = compiler.compile(source_code)

# Create kernel
kernel = CompiledKernel(
    lib_path=lib_path,
    function_name="custom_function",
    arg_types=[
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ],
    return_type=None
)

# Use the kernel
# ... (example usage)

# Clean up
kernel.cleanup()
```

## Architecture-Specific Optimizations

### ARM Architecture (NEON/SVE)

For ARM processors (including Apple Silicon and AWS Graviton), the backend generates code using:

- **NEON SIMD Instructions**: 128-bit SIMD operations, supported by all ARMv8+ processors
- **SVE/SVE2 Instructions**: Scalable Vector Extension, available on newer ARMv8.2+ and ARMv9 processors

Example of NEON vector addition:
```c
// NEON vector addition
for (i = 0; i <= n - 4; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(&c[i], vc);
}
```

### x86_64 Architecture (SSE/AVX/AVX-512)

For x86_64 processors (Intel and AMD), the backend generates code using:

- **SSE Instructions**: 128-bit SIMD operations
- **AVX/AVX2 Instructions**: 256-bit SIMD operations
- **AVX-512 Instructions**: 512-bit SIMD operations (on newer Intel CPUs)

Example of AVX vector addition:
```c
// AVX vector addition
for (i = 0; i <= n - 8; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(&c[i], vc);
}
```

## Compiler Optimization Flags

The backend automatically selects appropriate compiler optimization flags:

### GCC/Clang Flags
- `-O3`: Aggressive optimization
- `-march=native`: Optimize for the host CPU
- `-mfpu=neon`, `-mavx`, `-mavx2`, `-mavx512f`: Architecture-specific flags
- `-ffast-math`: Faster but slightly less precise math operations
- `-mfma`: FMA (Fused Multiply-Add) instructions

### MSVC Flags
- `/Ox`: Full optimization
- `/arch:AVX`, `/arch:AVX2`, `/arch:AVX512`: Architecture-specific flags
- `/std:c++17`, `/std:c11`: Language standards

## Error Handling

The backend includes robust error handling with specific exception types:

- `CompilerError`: Raised for errors during compilation
- `CRuntimeError`: Raised for errors during runtime execution

All exceptions include detailed error messages to aid in debugging.

## Performance Considerations

### When to Use the C/C++ Backend

The C/C++ backend is particularly useful in these scenarios:

1. **Portability with Performance**: When you need good performance across multiple platforms
2. **Complex Operations**: When implementing complex operations that would be difficult in assembly
3. **Compiler Optimization**: When you want to leverage compiler optimizations

### Optimization Tips

1. **Use Appropriate Data Types**: Stick with supported types (float32, float64, int32)
2. **Contiguous Memory**: Ensure input arrays are contiguous for optimal performance
3. **Large Workloads**: The C/C++ backend provides the biggest advantage for large workloads
4. **Clean Up Resources**: Always call `cleanup()` to release resources

## Advanced Topics

### Custom Operations

To implement a custom operation:

1. Create a specialized code generator method for your operation
2. Implement corresponding backend methods
3. Define appropriate ctypes signatures for your function
4. Register the operation with the backend registry

### Cross-Platform Considerations

- The backend automatically detects the appropriate compiler and flags
- The generated code includes platform-specific preprocessor directives
- For maximum portability, the backend includes fallback code paths

## Troubleshooting

Common issues and solutions:

1. **Compiler Not Found**: Ensure GCC, Clang, or MSVC is installed and in your PATH
2. **Compilation Errors**: Check that the generated code is valid for your platform
3. **Runtime Issues**: Ensure the function signatures match between C/C++ and Python
4. **Performance Issues**: Check that SIMD instructions are being used (enable log output)