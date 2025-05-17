"""
C/C++ code generation for the Prism framework.

This module provides classes for generating optimized C/C++ code
for different processor architectures.
"""

import os
import platform
import re
import textwrap
from typing import Dict, List, Optional, Tuple, Union, Any

from ...core.capability import get_device_capabilities


class CCodeGenerator:
    """
    Generator for C/C++ code optimized for different architectures.
    
    This class generates architecture-specific C/C++ code with optimizations
    for various instruction sets (NEON, SVE, AVX, AVX-512, etc.).
    """
    
    def __init__(self, 
                 arch: str = None, 
                 cpp: bool = True,
                 use_simd: bool = True):
        """
        Initialize the C/C++ code generator.
        
        Args:
            arch: Target architecture (auto-detected if None)
            cpp: Whether to generate C++ (True) or C (False) code
            use_simd: Whether to use SIMD instructions
        """
        self.arch = arch or self._detect_architecture()
        self.cpp = cpp
        self.use_simd = use_simd
        self.capabilities = get_device_capabilities()
        
        # Initialize architecture-specific code generation
        self._init_arch_specific()
    
    def _detect_architecture(self) -> str:
        """
        Detect the current system architecture.
        
        Returns:
            Architecture identifier
        """
        machine = platform.machine().lower()
        if machine in ("arm", "aarch64", "arm64"):
            return "arm64"
        elif machine in ("x86_64", "amd64", "x64"):
            return "x86_64"
        else:
            raise ValueError(f"Unsupported architecture: {machine}")
    
    def _init_arch_specific(self):
        """Initialize architecture-specific code generation."""
        self.simd_headers = []
        self.simd_macros = []
        
        if self.arch == "arm64":
            # ARM SIMD (NEON/SVE)
            if self.use_simd:
                self.simd_headers.append("#include <arm_neon.h>")
                
                # Check for SVE support
                if self.capabilities.get("sve_supported", False):
                    self.simd_headers.append("#include <arm_sve.h>")
                    
        elif self.arch == "x86_64":
            # x86_64 SIMD (SSE/AVX/AVX-512)
            if self.use_simd:
                self.simd_headers.extend([
                    "#include <immintrin.h>",
                    "#include <emmintrin.h>"
                ])
                
                # Add defines based on available instruction sets
                if self.capabilities.get("avx512_supported", False):
                    self.simd_macros.append("#define USE_AVX512")
                elif self.capabilities.get("avx2_supported", False):
                    self.simd_macros.append("#define USE_AVX2")
                elif self.capabilities.get("avx_supported", False):
                    self.simd_macros.append("#define USE_AVX")
    
    def generate_headers(self) -> str:
        """
        Generate header includes and macro definitions.
        
        Returns:
            String containing header includes and macros
        """
        headers = []
        
        # Standard headers
        headers.append("#include <stdio.h>")
        headers.append("#include <stdlib.h>")
        headers.append("#include <string.h>")
        headers.append("#include <math.h>")
        
        if self.cpp:
            # C++ specific headers
            headers.append("#include <cstdint>")
            headers.append("#include <algorithm>")
            headers.append("#include <vector>")
        else:
            # C specific headers
            headers.append("#include <stdint.h>")
        
        # Add architecture-specific headers
        headers.extend(self.simd_headers)
        
        # Add newline
        headers.append("")
        
        # Add architecture-specific macros
        headers.extend(self.simd_macros)
        
        if self.simd_macros:
            headers.append("")
        
        return "\n".join(headers)
    
    def generate_vector_add_function(self, 
                                    function_name: str = "vector_add",
                                    data_type: str = "float") -> str:
        """
        Generate C/C++ code for vector addition.
        
        Args:
            function_name: Name of the function
            data_type: Data type of the vectors
        
        Returns:
            C/C++ function for vector addition
        """
        if self.arch == "arm64" and self.use_simd:
            return self._generate_arm_vector_add(function_name, data_type)
        elif self.arch == "x86_64" and self.use_simd:
            return self._generate_x86_vector_add(function_name, data_type)
        else:
            # Generic implementation
            return self._generate_generic_vector_add(function_name, data_type)
    
    def _generate_arm_vector_add(self, function_name: str, data_type: str) -> str:
        """Generate ARM NEON/SVE optimized vector addition."""
        if data_type != "float":
            # Only float is supported for now
            return self._generate_generic_vector_add(function_name, data_type)
        
        if self.capabilities.get("sve_supported", False):
            # SVE implementation
            code = f"""
            void {function_name}(const float* a, const float* b, float* c, int n) {{
                // SVE implementation
                for (int i = 0; i < n; i += svcntw()) {{
                    // Create predicate for remaining elements
                    svbool_t pg = svwhilelt_b32(i, n);
                    
                    // Load vectors
                    svfloat32_t va = svld1_f32(pg, &a[i]);
                    svfloat32_t vb = svld1_f32(pg, &b[i]);
                    
                    // Add vectors
                    svfloat32_t vc = svadd_f32_m(pg, va, vb);
                    
                    // Store result
                    svst1_f32(pg, &c[i], vc);
                }}
            }}
            """
        else:
            # NEON implementation
            code = f"""
            void {function_name}(const float* a, const float* b, float* c, int n) {{
                // NEON implementation
                int i;
                
                // Process 4 elements at a time
                for (i = 0; i <= n - 4; i += 4) {{
                    // Load vectors
                    float32x4_t va = vld1q_f32(&a[i]);
                    float32x4_t vb = vld1q_f32(&b[i]);
                    
                    // Add vectors
                    float32x4_t vc = vaddq_f32(va, vb);
                    
                    // Store result
                    vst1q_f32(&c[i], vc);
                }}
                
                // Handle remaining elements
                for (; i < n; i++) {{
                    c[i] = a[i] + b[i];
                }}
            }}
            """
        
        return textwrap.dedent(code).strip()
    
    def _generate_x86_vector_add(self, function_name: str, data_type: str) -> str:
        """Generate x86_64 SSE/AVX/AVX-512 optimized vector addition."""
        if data_type != "float":
            # Only float is supported for now
            return self._generate_generic_vector_add(function_name, data_type)
        
        # Build with conditional compilation based on available instruction sets
        code = f"""
        void {function_name}(const float* a, const float* b, float* c, int n) {{
            int i;
            
        #ifdef USE_AVX512
            // AVX-512 implementation (16 elements at a time)
            for (i = 0; i <= n - 16; i += 16) {{
                // Load vectors
                __m512 va = _mm512_loadu_ps(&a[i]);
                __m512 vb = _mm512_loadu_ps(&b[i]);
                
                // Add vectors
                __m512 vc = _mm512_add_ps(va, vb);
                
                // Store result
                _mm512_storeu_ps(&c[i], vc);
            }}
        #elif defined(USE_AVX2) || defined(USE_AVX)
            // AVX/AVX2 implementation (8 elements at a time)
            for (i = 0; i <= n - 8; i += 8) {{
                // Load vectors
                __m256 va = _mm256_loadu_ps(&a[i]);
                __m256 vb = _mm256_loadu_ps(&b[i]);
                
                // Add vectors
                __m256 vc = _mm256_add_ps(va, vb);
                
                // Store result
                _mm256_storeu_ps(&c[i], vc);
            }}
        #else
            // SSE implementation (4 elements at a time)
            for (i = 0; i <= n - 4; i += 4) {{
                // Load vectors
                __m128 va = _mm_loadu_ps(&a[i]);
                __m128 vb = _mm_loadu_ps(&b[i]);
                
                // Add vectors
                __m128 vc = _mm_add_ps(va, vb);
                
                // Store result
                _mm_storeu_ps(&c[i], vc);
            }}
        #endif
            
            // Handle remaining elements
            for (; i < n; i++) {{
                c[i] = a[i] + b[i];
            }}
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def _generate_generic_vector_add(self, function_name: str, data_type: str) -> str:
        """Generate generic vector addition without SIMD."""
        code = f"""
        void {function_name}(const {data_type}* a, const {data_type}* b, {data_type}* c, int n) {{
            for (int i = 0; i < n; i++) {{
                c[i] = a[i] + b[i];
            }}
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def generate_matrix_multiply_function(self, 
                                         function_name: str = "matrix_multiply",
                                         data_type: str = "float") -> str:
        """
        Generate C/C++ code for matrix multiplication.
        
        Args:
            function_name: Name of the function
            data_type: Data type of the matrices
        
        Returns:
            C/C++ function for matrix multiplication
        """
        if self.arch == "arm64" and self.use_simd:
            return self._generate_arm_matrix_multiply(function_name, data_type)
        elif self.arch == "x86_64" and self.use_simd:
            return self._generate_x86_matrix_multiply(function_name, data_type)
        else:
            # Generic implementation
            return self._generate_generic_matrix_multiply(function_name, data_type)
    
    def _generate_arm_matrix_multiply(self, function_name: str, data_type: str) -> str:
        """Generate ARM NEON/SVE optimized matrix multiplication."""
        if data_type != "float":
            # Only float is supported for now
            return self._generate_generic_matrix_multiply(function_name, data_type)
        
        # NEON implementation (SVE is more complex for matrix multiplication)
        code = f"""
        void {function_name}(const float* a, const float* b, float* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof(float));
            
            // For each row of a
            for (int i = 0; i < m; i++) {{
                // For each column of b
                for (int j = 0; j < n; j++) {{
                    // For blocks of k
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int kb;
                    
                    // Process 4 elements at a time
                    for (kb = 0; kb <= k - 4; kb += 4) {{
                        // Load 4 elements from a and 4 elements from b
                        float32x4_t a_vec = vld1q_f32(&a[i * k + kb]);
                        
                        // Load 4 elements from column j of b (with stride n)
                        float32_t b_array[4];
                        for (int l = 0; l < 4; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        float32x4_t b_vec = vld1q_f32(b_array);
                        
                        // Multiply and accumulate
                        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                    }}
                    
                    // Reduce the vector sum to a scalar
                    float sum = vgetq_lane_f32(sum_vec, 0) + 
                                vgetq_lane_f32(sum_vec, 1) + 
                                vgetq_lane_f32(sum_vec, 2) + 
                                vgetq_lane_f32(sum_vec, 3);
                    
                    // Handle remaining elements
                    for (; kb < k; kb++) {{
                        sum += a[i * k + kb] * b[kb * n + j];
                    }}
                    
                    // Store the result
                    c[i * n + j] = sum;
                }}
            }}
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def _generate_x86_matrix_multiply(self, function_name: str, data_type: str) -> str:
        """Generate x86_64 SSE/AVX/AVX-512 optimized matrix multiplication."""
        if data_type != "float":
            # Only float is supported for now
            return self._generate_generic_matrix_multiply(function_name, data_type)
        
        # Build with conditional compilation based on available instruction sets
        code = f"""
        void {function_name}(const float* a, const float* b, float* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof(float));
            
        #if defined(USE_AVX512)
            // Use AVX-512 if available (most efficient)
            const int block_size = 16;  // Process 16 elements at a time
        #elif defined(USE_AVX2) || defined(USE_AVX)
            // Use AVX/AVX2 if available
            const int block_size = 8;   // Process 8 elements at a time
        #else
            // Fall back to SSE
            const int block_size = 4;   // Process 4 elements at a time
        #endif
            
            // Block-based matrix multiplication for better cache utilization
            for (int i = 0; i < m; i++) {{
                for (int j = 0; j < n; j++) {{
                    float sum = 0.0f;
                    
                    // Process blocks of k
                    int kb;
                    for (kb = 0; kb <= k - block_size; kb += block_size) {{
        #if defined(USE_AVX512)
                        // AVX-512 implementation
                        __m512 sum_vec = _mm512_setzero_ps();
                        __m512 a_vec = _mm512_loadu_ps(&a[i * k + kb]);
                        
                        // Load 16 elements from column j of b (with stride n)
                        float b_array[16];
                        for (int l = 0; l < 16; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        __m512 b_vec = _mm512_loadu_ps(b_array);
                        
                        // Multiply and accumulate
                        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
                        
                        // Reduce the vector sum to a scalar
                        sum += _mm512_reduce_add_ps(sum_vec);
        #elif defined(USE_AVX2)
                        // AVX2 implementation with FMA
                        __m256 sum_vec = _mm256_setzero_ps();
                        __m256 a_vec = _mm256_loadu_ps(&a[i * k + kb]);
                        
                        // Load 8 elements from column j of b (with stride n)
                        float b_array[8];
                        for (int l = 0; l < 8; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        __m256 b_vec = _mm256_loadu_ps(b_array);
                        
                        // Multiply and accumulate using FMA
                        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        
                        // Reduce the vector sum to a scalar
                        float sum_array[8];
                        _mm256_storeu_ps(sum_array, sum_vec);
                        for (int l = 0; l < 8; l++) {{
                            sum += sum_array[l];
                        }}
        #elif defined(USE_AVX)
                        // AVX implementation
                        __m256 sum_vec = _mm256_setzero_ps();
                        __m256 a_vec = _mm256_loadu_ps(&a[i * k + kb]);
                        
                        // Load 8 elements from column j of b (with stride n)
                        float b_array[8];
                        for (int l = 0; l < 8; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        __m256 b_vec = _mm256_loadu_ps(b_array);
                        
                        // Multiply and accumulate
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a_vec, b_vec));
                        
                        // Reduce the vector sum to a scalar
                        float sum_array[8];
                        _mm256_storeu_ps(sum_array, sum_vec);
                        for (int l = 0; l < 8; l++) {{
                            sum += sum_array[l];
                        }}
        #else
                        // SSE implementation
                        __m128 sum_vec = _mm_setzero_ps();
                        __m128 a_vec = _mm_loadu_ps(&a[i * k + kb]);
                        
                        // Load 4 elements from column j of b (with stride n)
                        float b_array[4];
                        for (int l = 0; l < 4; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        __m128 b_vec = _mm_loadu_ps(b_array);
                        
                        // Multiply and accumulate
                        sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(a_vec, b_vec));
                        
                        // Reduce the vector sum to a scalar
                        float sum_array[4];
                        _mm_storeu_ps(sum_array, sum_vec);
                        sum += sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        #endif
                    }}
                    
                    // Handle remaining elements
                    for (; kb < k; kb++) {{
                        sum += a[i * k + kb] * b[kb * n + j];
                    }}
                    
                    // Store the result
                    c[i * n + j] = sum;
                }}
            }}
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def _generate_generic_matrix_multiply(self, function_name: str, data_type: str) -> str:
        """Generate generic matrix multiplication without SIMD."""
        code = f"""
        void {function_name}(const {data_type}* a, const {data_type}* b, {data_type}* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof({data_type}));
            
            // Loop over rows of a
            for (int i = 0; i < m; i++) {{
                // Loop over columns of b
                for (int j = 0; j < n; j++) {{
                    // Compute dot product of row i of a and column j of b
                    {data_type} sum = 0;
                    for (int l = 0; l < k; l++) {{
                        sum += a[i * k + l] * b[l * n + j];
                    }}
                    c[i * n + j] = sum;
                }}
            }}
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def generate_full_source(self, 
                           function_name: str, 
                           operation: str, 
                           data_type: str = "float",
                           with_main: bool = False) -> str:
        """
        Generate complete C/C++ source code for an operation.
        
        Args:
            function_name: Name of the function
            operation: Operation type ("vector_add" or "matrix_multiply")
            data_type: Data type to use
            with_main: Whether to include a main function
        
        Returns:
            Complete C/C++ source code
        """
        # Generate headers
        source = self.generate_headers()
        
        # Add export directive for Windows
        source += "\n#ifdef _WIN32\n"
        source += "#define EXPORT __declspec(dllexport)\n"
        source += "#else\n"
        source += "#define EXPORT\n"
        source += "#endif\n\n"
        
        # Generate function based on operation
        if operation == "vector_add":
            function_code = self.generate_vector_add_function(function_name, data_type)
            source += "EXPORT " + function_code
        elif operation == "matrix_multiply":
            function_code = self.generate_matrix_multiply_function(function_name, data_type)
            source += "EXPORT " + function_code
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        # Add main function if requested
        if with_main:
            if operation == "vector_add":
                main = self._generate_vector_add_main(function_name, data_type)
            elif operation == "matrix_multiply":
                main = self._generate_matrix_multiply_main(function_name, data_type)
            else:
                main = "int main() { return 0; }"
            
            source += f"\n\n{main}"
        
        return source
    
    def _generate_vector_add_main(self, function_name: str, data_type: str) -> str:
        """Generate a main function for testing vector addition."""
        code = f"""
        int main() {{
            const int n = 1000;
            {data_type}* a = ({data_type}*)malloc(n * sizeof({data_type}));
            {data_type}* b = ({data_type}*)malloc(n * sizeof({data_type}));
            {data_type}* c = ({data_type}*)malloc(n * sizeof({data_type}));
            
            // Initialize arrays
            for (int i = 0; i < n; i++) {{
                a[i] = ({data_type})i / 10.0;
                b[i] = ({data_type})i / 20.0;
            }}
            
            // Call vector addition
            {function_name}(a, b, c, n);
            
            // Print some results
            printf("Results:\\n");
            for (int i = 0; i < 10; i++) {{
                printf("%d: %.2f + %.2f = %.2f\\n", i, a[i], b[i], c[i]);
            }}
            
            // Clean up
            free(a);
            free(b);
            free(c);
            
            return 0;
        }}
        """
        
        return textwrap.dedent(code).strip()
    
    def _generate_matrix_multiply_main(self, function_name: str, data_type: str) -> str:
        """Generate a main function for testing matrix multiplication."""
        code = f"""
        int main() {{
            const int m = 10;
            const int n = 10;
            const int k = 10;
            
            {data_type}* a = ({data_type}*)malloc(m * k * sizeof({data_type}));
            {data_type}* b = ({data_type}*)malloc(k * n * sizeof({data_type}));
            {data_type}* c = ({data_type}*)malloc(m * n * sizeof({data_type}));
            
            // Initialize matrices
            for (int i = 0; i < m; i++) {{
                for (int j = 0; j < k; j++) {{
                    a[i * k + j] = ({data_type})(i + j) / 10.0;
                }}
            }}
            
            for (int i = 0; i < k; i++) {{
                for (int j = 0; j < n; j++) {{
                    b[i * n + j] = ({data_type})(i - j) / 10.0;
                }}
            }}
            
            // Call matrix multiplication
            {function_name}(a, b, c, m, n, k);
            
            // Print some results
            printf("Results:\\n");
            for (int i = 0; i < 3; i++) {{
                for (int j = 0; j < 3; j++) {{
                    printf("%.2f ", c[i * n + j]);
                }}
                printf("\\n");
            }}
            
            // Clean up
            free(a);
            free(b);
            free(c);
            
            return 0;
        }}
        """
        
        return textwrap.dedent(code).strip()