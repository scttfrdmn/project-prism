"""
Integration module for low-level backends with the core Prism framework.

This module provides the glue code to connect low-level backends (C/C++, Assembly)
with the higher-level Prism framework, allowing seamless execution of operations
using architecture-specific optimized code.
"""

import os
import sys
import logging
import inspect
import platform
import importlib
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from ..capability import get_device_capabilities
from ..operation import Operation, OperationGraph
from ...backends.asmbackend import AssemblyBackend, get_examples

logger = logging.getLogger(__name__)


class LowLevelBackendRegistry:
    """Registry for low-level backend implementations."""
    
    def __init__(self):
        """Initialize the low-level backend registry."""
        self.backends = {}
        self.operation_mappings = {}
    
    def register_backend(self, backend_name: str, backend_class: Any) -> None:
        """
        Register a low-level backend.
        
        Args:
            backend_name: Name of the backend
            backend_class: Backend class or factory function
        """
        self.backends[backend_name] = backend_class
        self.operation_mappings[backend_name] = {}
    
    def register_operation(self, backend_name: str, operation_name: str, 
                          implementation: Callable) -> None:
        """
        Register an operation implementation for a specific backend.
        
        Args:
            backend_name: Name of the backend
            operation_name: Name of the operation
            implementation: Implementation function or class
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not registered")
        
        self.operation_mappings[backend_name][operation_name] = implementation
    
    def get_backend(self, backend_name: str) -> Any:
        """
        Get a backend by name.
        
        Args:
            backend_name: Name of the backend
        
        Returns:
            Backend instance
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not registered")
        
        # If the backend is a class, instantiate it
        backend = self.backends[backend_name]
        if inspect.isclass(backend):
            return backend()
        return backend
    
    def get_operation_implementation(self, backend_name: str, operation_name: str) -> Callable:
        """
        Get operation implementation for a specific backend.
        
        Args:
            backend_name: Name of the backend
            operation_name: Name of the operation
        
        Returns:
            Operation implementation
        """
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not registered")
        
        if operation_name not in self.operation_mappings[backend_name]:
            raise ValueError(f"Operation '{operation_name}' not implemented for backend '{backend_name}'")
        
        return self.operation_mappings[backend_name][operation_name]
    
    def has_operation(self, backend_name: str, operation_name: str) -> bool:
        """
        Check if an operation is implemented for a specific backend.
        
        Args:
            backend_name: Name of the backend
            operation_name: Name of the operation
        
        Returns:
            True if the operation is implemented, False otherwise
        """
        if backend_name not in self.backends:
            return False
        
        return operation_name in self.operation_mappings[backend_name]


# Create a global registry instance
registry = LowLevelBackendRegistry()


class AssemblyBackendIntegration:
    """Integration class for the Assembly backend."""
    
    def __init__(self):
        """Initialize the Assembly backend integration."""
        self.backend = AssemblyBackend()
        self.examples = get_examples(self.backend.arch, self.backend.syntax)
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform vector addition using optimized assembly code.
        
        Args:
            a: First input vector (float32)
            b: Second input vector (float32)
        
        Returns:
            Result vector (float32)
        """
        import ctypes
        
        # Ensure inputs are float32
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        
        # Check shapes
        if a.shape != b.shape:
            raise ValueError(f"Input shapes must match: {a.shape} vs {b.shape}")
        
        # Create output array
        result = np.zeros_like(a)
        size = np.prod(a.shape)
        
        # Generate assembly code
        if self.backend.arch == "arm64":
            asm_code = self.examples.vector_add_float32("a", "b", "c", size)
        else:
            asm_code = self.examples.vector_add_float32("a", "b", "c", size, 
                                                       syntax=self.backend.syntax)
        
        # Compile and execute
        try:
            # Create contiguous flattened arrays
            a_flat = np.ascontiguousarray(a.flatten())
            b_flat = np.ascontiguousarray(b.flatten())
            result_flat = np.ascontiguousarray(result.flatten())
            
            # Create ctypes pointers
            a_ptr = a_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result_ptr = result_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Execute assembly function
            self.backend.execute(
                asm_code=asm_code,
                function_name="vector_add_float32",
                args=[a_ptr, b_ptr, result_ptr, ctypes.c_int(size)],
                arg_types=[
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int
                ],
                return_type=None,
                cache=True  # Cache the compiled kernel
            )
            
            # Reshape result to original shape
            return result_flat.reshape(a.shape)
            
        except Exception as e:
            # Fall back to numpy if assembly execution fails
            logger.warning(f"Assembly backend failed, falling back to numpy: {str(e)}")
            return a + b
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication using optimized assembly code.
        
        Args:
            a: First input matrix (float32)
            b: Second input matrix (float32)
        
        Returns:
            Result matrix (float32)
        """
        import ctypes
        
        # Ensure inputs are float32
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        
        # Check shapes
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible matrix shapes for multiplication: {a.shape} and {b.shape}")
        
        # Extract dimensions
        m, k = a.shape
        k2, n = b.shape
        
        # Create output array
        result = np.zeros((m, n), dtype=np.float32)
        
        # Generate assembly code
        if self.backend.arch == "arm64":
            asm_code = self.examples.matrix_multiply_float32("a", "b", "c", m, n, k)
        else:
            asm_code = self.examples.matrix_multiply_float32("a", "b", "c", m, n, k, 
                                                           syntax=self.backend.syntax)
        
        # Compile and execute
        try:
            # Create contiguous arrays
            a_cont = np.ascontiguousarray(a)
            b_cont = np.ascontiguousarray(b)
            result_cont = np.ascontiguousarray(result)
            
            # Create ctypes pointers
            a_ptr = a_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result_ptr = result_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Execute assembly function
            self.backend.execute(
                asm_code=asm_code,
                function_name="matrix_multiply_float32",
                args=[a_ptr, b_ptr, result_ptr, ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k)],
                arg_types=[
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int
                ],
                return_type=None,
                cache=True  # Cache the compiled kernel
            )
            
            return result_cont
            
        except Exception as e:
            # Fall back to numpy if assembly execution fails
            logger.warning(f"Assembly backend failed, falling back to numpy: {str(e)}")
            return np.matmul(a, b)
    
    def cleanup(self):
        """Clean up resources used by the backend."""
        if hasattr(self, 'backend'):
            self.backend.cleanup()


class CBackendIntegration:
    """Integration class for the C/C++ backend."""
    
    def __init__(self):
        """Initialize the C/C++ backend integration."""
        # Import the C backend lazily to avoid circular imports
        try:
            from ...backends.cbackend import CBackend
            self.backend = CBackend()
        except ImportError:
            self.backend = None
            logger.warning("C backend not available")
    
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform vector addition using optimized C/C++ code.
        
        Args:
            a: First input vector (float32)
            b: Second input vector (float32)
        
        Returns:
            Result vector (float32)
        """
        if self.backend is None:
            return a + b  # Fall back to numpy
        
        # Placeholder for C backend implementation
        # This will be implemented once the C backend is complete
        return a + b
    
    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform matrix multiplication using optimized C/C++ code.
        
        Args:
            a: First input matrix (float32)
            b: Second input matrix (float32)
        
        Returns:
            Result matrix (float32)
        """
        if self.backend is None:
            return np.matmul(a, b)  # Fall back to numpy
        
        # Placeholder for C backend implementation
        # This will be implemented once the C backend is complete
        return np.matmul(a, b)
    
    def cleanup(self):
        """Clean up resources used by the backend."""
        if hasattr(self, 'backend') and self.backend is not None:
            if hasattr(self.backend, 'cleanup'):
                self.backend.cleanup()


# Register backends
registry.register_backend("assembly", AssemblyBackendIntegration)
registry.register_backend("c", CBackendIntegration)

# Register operations for assembly backend
registry.register_operation("assembly", "vector_add", lambda instance: instance.vector_add)
registry.register_operation("assembly", "matrix_multiply", lambda instance: instance.matrix_multiply)

# Register operations for C backend
registry.register_operation("c", "vector_add", lambda instance: instance.vector_add)
registry.register_operation("c", "matrix_multiply", lambda instance: instance.matrix_multiply)


def get_optimal_backend_for_operation(operation_name: str) -> str:
    """
    Determine the optimal backend for a given operation based on hardware capabilities.
    
    Args:
        operation_name: Name of the operation
    
    Returns:
        Name of the optimal backend
    """
    # Get device capabilities
    capabilities = get_device_capabilities()
    architecture = platform.machine().lower()
    
    # Determine available backends for the operation
    available_backends = [
        name for name in registry.backends
        if registry.has_operation(name, operation_name)
    ]
    
    if not available_backends:
        raise ValueError(f"No backend available for operation: {operation_name}")
    
    # Decision logic based on operation and hardware
    if operation_name in ["vector_add", "matrix_multiply"]:
        # For vector and matrix operations, prefer assembly on modern processors
        if architecture in ("arm64", "aarch64"):
            # Check for ARM-specific features
            if capabilities.get("sve_supported", False):
                # SVE is great for vector operations
                return "assembly" if "assembly" in available_backends else "c"
            elif capabilities.get("neon_supported", False):
                # NEON is also good for vector operations
                return "assembly" if "assembly" in available_backends else "c"
        
        elif architecture in ("x86_64", "amd64"):
            # Check for x86_64-specific features
            if capabilities.get("avx512_supported", False):
                # AVX-512 is excellent for vector operations
                return "assembly" if "assembly" in available_backends else "c"
            elif capabilities.get("avx2_supported", False):
                # AVX2 is very good for vector operations
                return "assembly" if "assembly" in available_backends else "c"
            elif capabilities.get("avx_supported", False):
                # AVX is good for vector operations
                return "assembly" if "assembly" in available_backends else "c"
    
    # Default to the first available backend
    return available_backends[0]


def execute_operation(operation_name: str, *args, **kwargs) -> Any:
    """
    Execute an operation using the optimal backend.
    
    Args:
        operation_name: Name of the operation
        *args: Positional arguments for the operation
        **kwargs: Keyword arguments for the operation
    
    Returns:
        Result of the operation
    """
    # Get the optimal backend
    backend_name = kwargs.pop("backend", get_optimal_backend_for_operation(operation_name))
    
    # Get the backend instance
    backend_instance = registry.get_backend(backend_name)
    
    # Get the operation implementation
    implementation_getter = registry.get_operation_implementation(backend_name, operation_name)
    implementation = implementation_getter(backend_instance)
    
    # Execute the operation
    try:
        result = implementation(*args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Error executing operation {operation_name} with backend {backend_name}: {str(e)}")
        raise
    finally:
        # Clean up resources if needed
        if hasattr(backend_instance, 'cleanup'):
            backend_instance.cleanup()


# Convenience functions for common operations
def vector_add(a: np.ndarray, b: np.ndarray, backend: str = None) -> np.ndarray:
    """
    Add two vectors using the optimal backend.
    
    Args:
        a: First input vector
        b: Second input vector
        backend: Optional backend name to use
    
    Returns:
        Result vector
    """
    kwargs = {"backend": backend} if backend else {}
    return execute_operation("vector_add", a, b, **kwargs)


def matrix_multiply(a: np.ndarray, b: np.ndarray, backend: str = None) -> np.ndarray:
    """
    Multiply two matrices using the optimal backend.
    
    Args:
        a: First input matrix
        b: Second input matrix
        backend: Optional backend name to use
    
    Returns:
        Result matrix
    """
    kwargs = {"backend": backend} if backend else {}
    return execute_operation("matrix_multiply", a, b, **kwargs)