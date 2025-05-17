# Compiler Support in Prism Framework

The Prism framework provides support for multiple compilers to enable optimal code generation and execution across different architectures and platforms. This document outlines the supported compilers and their specific optimizations.

## Supported Compilers

The C/C++ backend in Prism supports the following compilers:

### 1. GCC (GNU Compiler Collection)
- **Detection**: Automatically detected in PATH as `gcc`/`g++`
- **Platforms**: Linux, macOS, Windows (MinGW/Cygwin)
- **Optimizations**: 
  - Native architecture optimizations (`-march=native`)
  - SIMD instruction sets (SSE, AVX, AVX2, AVX-512)
  - Fast math operations (`-ffast-math`)
  - FMA instructions (`-mfma`)
  - Tree vectorization (`-ftree-vectorize`)

### 2. Clang/LLVM
- **Detection**: Automatically detected in PATH as `clang`/`clang++`
- **Platforms**: Linux, macOS, Windows
- **Optimizations**:
  - Native architecture optimizations (`-march=native`)
  - SIMD instruction sets (SSE, AVX, AVX2, AVX-512)
  - Fast math operations (`-ffast-math`)
  - FMA instructions (`-mfma`)

### 3. Intel oneAPI Compiler (formerly Intel C/C++ Compiler)
- **Detection**: 
  - Automatically detected in PATH as `icx`/`icpx`
  - Special paths checked: `/opt/intel/oneapi/compiler/latest/bin/`
  - Automatic activation of oneAPI environment if available
- **Platforms**: Linux, macOS, Windows
- **Optimizations**:
  - Host-specific optimizations (`-xHost`)
  - SIMD instruction sets (SSE, AVX, AVX2, AVX-512)
  - Interprocedural optimization (`-ipo`)
  - Memory layout optimizations (`-qopt-mem-layout-trans=3`)
  - AVX-512 register optimization (`-qopt-zmm-usage=high`)
  - Fast math operations (`-ffast-math`)

### 4. ARM Compiler
- **Detection**:
  - Automatically detected in PATH as `armclang`/`armclang++`
  - Special paths checked: `/opt/arm/arm-compiler/bin/`
- **Platforms**: Linux, macOS (Apple Silicon), Windows
- **Optimizations**:
  - ARM architecture specific flags (`-march=armv8-a`)
  - NEON SIMD instructions (`-mfpu=neon`)
  - SVE instructions (`-march=armv8.2-a+sve`)
  - Optimize for speed over size (`-Otime`)
  - Fast math operations (`-ffast-math`)
  - Floating-point operation contraction (`-ffp-contract=fast`)

### 5. AMD AOCC (AMD Optimizing C/C++ Compiler)
- **Detection**: 
  - Automatically detected in PATH as `aocl-clang`/`aocl-clang++`
  - Special paths checked: `/opt/AMD/aocc/bin/`
- **Platforms**: Linux, Windows
- **Optimizations**:
  - Native architecture optimizations (`-march=native`)
  - AMD FMA instructions (`-mafma`)
  - SIMD instruction sets (AVX, AVX2)
  - Fast math operations (`-ffast-math`)
  - Loop unrolling (`-funroll-loops`)
  - Tree vectorization (`-ftree-vectorize`)
  - Polly polyhedral optimizer (`-mllvm -polly`)
  - Link-time optimization (`-flto`)

### 6. Microsoft Visual C++ Compiler (MSVC)
- **Detection**: Automatically detected in PATH as `cl`
- **Platforms**: Windows
- **Optimizations**:
  - Full optimization (`/Ox`)
  - Architecture-specific (`/arch:AVX`, `/arch:AVX2`, `/arch:AVX512`, `/arch:arm64`)
  - C++17/C11 standards (`/std:c++17`, `/std:c11`)

## Compiler Detection and Selection

The framework follows these steps to select a compiler:

1. **Automatic Detection**:
   - Searches for compilers in the system PATH
   - Checks special installation directories for specialized compilers
   - Attempts to activate vendor-specific environments (e.g., Intel oneAPI)

2. **Preference Order**:
   - On macOS: clang++, g++, icpx, armclang++ (for C++)
   - On Windows: cl, g++, clang++, icpx, armclang++, aocl-clang++ (for C++)
   - On Linux: g++, clang++, icpx, armclang++, aocl-clang++ (for C++)

3. **Compiler Type Detection**:
   - Analyzes the compiler path to determine the compiler type
   - Applies appropriate flags and optimizations based on the compiler type

## Architecture-Specific Optimizations

### ARM Architecture (arm64)
- NEON SIMD instructions for 128-bit vector operations
- SVE (Scalable Vector Extension) for newer ARM processors
- Specific optimizations for Apple Silicon and AWS Graviton

### x86_64 Architecture
- SSE instructions for 128-bit vector operations
- AVX/AVX2 instructions for 256-bit vector operations 
- AVX-512 instructions for 512-bit vector operations
- FMA (Fused Multiply-Add) instructions for faster multiply-add operations

## Compiler Environment Configuration

### Intel oneAPI
The framework will automatically attempt to source the Intel oneAPI environment script (`/opt/intel/oneapi/setvars.sh`) if it exists, enabling access to the Intel compilers even if they are not directly in the PATH.

### Custom Compiler Paths
You can specify a custom compiler by directly instantiating the `CompilerWrapper` with a specific compiler path:

```python
from src.backends.cbackend import CompilerWrapper

# Use a specific compiler
compiler = CompilerWrapper(compiler="/path/to/custom/compiler")
```

## Adding New Compiler Support

To add support for a new compiler:

1. Update the `_detect_compiler` method in `CompilerWrapper` to search for the new compiler
2. Add the compiler to the `_detect_compiler_type` method to identify its type
3. Update the `_generate_arch_flags` method to include architecture-specific flags for the new compiler
4. Update the `_get_compiler_flags` method to include compiler-specific optimization flags

## Troubleshooting

If you encounter issues with compiler detection or compilation:

1. **Compiler Not Found**: Ensure the desired compiler is installed and in your PATH
2. **Compilation Errors**: Check that the generated code is compatible with your compiler version
3. **Missing Optimizations**: Review the compiler flags being used with the `get_compiler_version` method
4. **Environment Issues**: For vendor-specific compilers like Intel oneAPI, ensure the environment is properly set up