"""
Templates for common C/C++ operations.

This module provides pre-defined templates for common C/C++ operations
with optimizations for different architectures.
"""

import platform
from typing import Dict, Optional, Union, Any


class OperationTemplate:
    """Base class for operation templates."""
    
    def __init__(self, arch: str = None, data_type: str = "float"):
        """
        Initialize the operation template.
        
        Args:
            arch: Target architecture (auto-detected if None)
            data_type: Data type for the operation
        """
        self.arch = arch or self._detect_architecture()
        self.data_type = data_type
    
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
    
    def generate(self) -> str:
        """
        Generate C/C++ code for the operation.
        
        Returns:
            C/C++ code
        """
        raise NotImplementedError("Subclasses must implement this method")


class VectorAddTemplate(OperationTemplate):
    """Template for vector addition."""
    
    def generate(self) -> str:
        """
        Generate C/C++ code for vector addition.
        
        Returns:
            C/C++ code for vector addition
        """
        if self.arch == "arm64":
            return self._generate_arm_vector_add()
        elif self.arch == "x86_64":
            return self._generate_x86_vector_add()
        else:
            return self._generate_generic_vector_add()
    
    def _generate_arm_vector_add(self) -> str:
        """Generate ARM optimized vector addition."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <arm_neon.h>
        
        // Vector addition optimized for ARM architecture
        void vector_add(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int n) {{
            int i;
            
            // Process 4 elements at a time using NEON
            for (i = 0; i <= n - 4; i += 4) {{
                float32x4_t va = vld1q_f32(&a[i]);
                float32x4_t vb = vld1q_f32(&b[i]);
                float32x4_t vc = vaddq_f32(va, vb);
                vst1q_f32(&c[i], vc);
            }}
            
            // Handle remaining elements
            for (; i < n; i++) {{
                c[i] = a[i] + b[i];
            }}
        }}
        """
        return template
    
    def _generate_x86_vector_add(self) -> str:
        """Generate x86_64 optimized vector addition."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <immintrin.h>
        
        // Vector addition optimized for x86_64 architecture
        void vector_add(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int n) {{
            int i;
            
            // Check for AVX support at runtime
            #if defined(__AVX__)
            // Process 8 elements at a time using AVX
            for (i = 0; i <= n - 8; i += 8) {{
                __m256 va = _mm256_loadu_ps(&a[i]);
                __m256 vb = _mm256_loadu_ps(&b[i]);
                __m256 vc = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(&c[i], vc);
            }}
            #else
            // Process 4 elements at a time using SSE
            for (i = 0; i <= n - 4; i += 4) {{
                __m128 va = _mm_loadu_ps(&a[i]);
                __m128 vb = _mm_loadu_ps(&b[i]);
                __m128 vc = _mm_add_ps(va, vb);
                _mm_storeu_ps(&c[i], vc);
            }}
            #endif
            
            // Handle remaining elements
            for (; i < n; i++) {{
                c[i] = a[i] + b[i];
            }}
        }}
        """
        return template
    
    def _generate_generic_vector_add(self) -> str:
        """Generate generic vector addition."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        
        // Generic vector addition
        void vector_add(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int n) {{
            for (int i = 0; i < n; i++) {{
                c[i] = a[i] + b[i];
            }}
        }}
        """
        return template


class MatrixMultiplyTemplate(OperationTemplate):
    """Template for matrix multiplication."""
    
    def generate(self) -> str:
        """
        Generate C/C++ code for matrix multiplication.
        
        Returns:
            C/C++ code for matrix multiplication
        """
        if self.arch == "arm64":
            return self._generate_arm_matrix_multiply()
        elif self.arch == "x86_64":
            return self._generate_x86_matrix_multiply()
        else:
            return self._generate_generic_matrix_multiply()
    
    def _generate_arm_matrix_multiply(self) -> str:
        """Generate ARM optimized matrix multiplication."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <arm_neon.h>
        
        // Matrix multiplication optimized for ARM architecture
        void matrix_multiply(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof({self.data_type}));
            
            const int block_size = 4;  // NEON processes 4 elements at a time
            
            // Block-based matrix multiplication for better cache utilization
            for (int i = 0; i < m; i++) {{
                for (int j = 0; j < n; j++) {{
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int kb;
                    
                    // Process blocks of k
                    for (kb = 0; kb <= k - block_size; kb += block_size) {{
                        float32x4_t a_vec = vld1q_f32(&a[i * k + kb]);
                        
                        // Load 4 elements from column j of b (with stride n)
                        float b_array[4];
                        for (int l = 0; l < 4; l++) {{
                            b_array[l] = b[(kb + l) * n + j];
                        }}
                        float32x4_t b_vec = vld1q_f32(b_array);
                        
                        // Multiply and accumulate
                        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                    }}
                    
                    // Reduce the vector sum to a scalar
                    float sum_array[4];
                    vst1q_f32(sum_array, sum_vec);
                    float sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
                    
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
        return template
    
    def _generate_x86_matrix_multiply(self) -> str:
        """Generate x86_64 optimized matrix multiplication."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        #include <immintrin.h>
        
        // Matrix multiplication optimized for x86_64 architecture
        void matrix_multiply(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof({self.data_type}));
            
            // Block size depends on available instruction set
            #if defined(__AVX__)
            const int block_size =
            8;  // AVX processes 8 elements at a time
            #else
            const int block_size = 4;  // SSE processes 4 elements at a time
            #endif
            
            // Block-based matrix multiplication for better cache utilization
            for (int i = 0; i < m; i++) {{
                for (int j = 0; j < n; j++) {{
                    float sum = 0.0f;
                    
                    // Process blocks of k
                    int kb;
                    for (kb = 0; kb <= k - block_size; kb += block_size) {{
                        #if defined(__AVX__)
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
        return template
    
    def _generate_generic_matrix_multiply(self) -> str:
        """Generate generic matrix multiplication."""
        template = f"""
        #include <stdio.h>
        #include <stdlib.h>
        #include <string.h>
        
        // Generic matrix multiplication
        void matrix_multiply(const {self.data_type}* a, const {self.data_type}* b, {self.data_type}* c, int m, int n, int k) {{
            // a: m x k matrix
            // b: k x n matrix
            // c: m x n matrix
            
            // Zero the output matrix
            memset(c, 0, m * n * sizeof({self.data_type}));
            
            // Loop over rows of a
            for (int i = 0; i < m; i++) {{
                // Loop over columns of b
                for (int j = 0; j < n; j++) {{
                    // Compute dot product of row i of a and column j of b
                    {self.data_type} sum = 0;
                    for (int l = 0; l < k; l++) {{
                        sum += a[i * k + l] * b[l * n + j];
                    }}
                    c[i * n + j] = sum;
                }}
            }}
        }}
        """
        return template


# Dictionary of available templates
_TEMPLATES = {
    "vector_add": VectorAddTemplate,
    "matrix_multiply": MatrixMultiplyTemplate,
}


def get_template(operation: str, arch: str = None, data_type: str = "float") -> str:
    """
    Get a C/C++ code template for an operation.
    
    Args:
        operation: Operation name
        arch: Target architecture (auto-detected if None)
        data_type: Data type for the operation
    
    Returns:
        C/C++ code template
    
    Raises:
        ValueError: If the operation is not supported
    """
    if operation not in _TEMPLATES:
        raise ValueError(f"Unsupported operation: {operation}")
    
    template_class = _TEMPLATES[operation]
    template = template_class(arch=arch, data_type=data_type)
    return template.generate()