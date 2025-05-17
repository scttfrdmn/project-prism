"""
Runtime execution for C/C++ code in the Prism framework.

This module provides classes for loading and executing compiled C/C++
code as dynamically loaded libraries using ctypes.
"""

import os
import sys
import ctypes
import logging
import platform
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from .codegen import CCodeGenerator
from .compiler import CompilerWrapper, CompilerError

logger = logging.getLogger(__name__)


class CRuntimeError(Exception):
    """Exception raised for errors during C/C++ code execution."""
    pass


class CompiledKernel:
    """
    Represents a compiled C/C++ kernel that can be executed.
    
    This class handles loading the compiled C/C++ code as a shared library
    and providing a Python callable interface to the C/C++ functions.
    """
    
    def __init__(self, 
                 lib_path: str, 
                 function_name: str, 
                 arg_types: List[type], 
                 return_type: type = None,
                 own_lib: bool = True):
        """
        Initialize the compiled kernel.
        
        Args:
            lib_path: Path to the compiled library
            function_name: Name of the function in the C/C++ code
            arg_types: List of ctypes argument types
            return_type: Return type of the function (None for void)
            own_lib: Whether this instance owns the library file (will delete on cleanup)
        """
        self.lib_path = lib_path
        self.function_name = function_name
        self.arg_types = arg_types
        self.return_type = return_type
        self.own_lib = own_lib
        self.lib_handle = None
        self.function_handle = None
        
        # Load the library
        self._load_library()
    
    def _load_library(self) -> None:
        """Load the shared library and set up the function handle."""
        try:
            # Load the shared library
            if sys.platform == 'win32':
                # Windows has special handling for DLLs
                self.lib_handle = ctypes.WinDLL(self.lib_path)
            else:
                # Unix/Mac
                self.lib_handle = ctypes.CDLL(self.lib_path)
            
            # Get the function from the library
            try:
                func = getattr(self.lib_handle, self.function_name)
            except AttributeError:
                raise CRuntimeError(f"Function '{self.function_name}' not found in library '{self.lib_path}'")
            
            # Set the function argument and return types
            func.argtypes = self.arg_types
            func.restype = self.return_type
            
            # Store the function handle
            self.function_handle = func
            
        except (OSError, AttributeError) as e:
            raise CRuntimeError(f"Failed to load the shared library: {str(e)}")
    
    def __call__(self, *args):
        """
        Call the C/C++ function with the given arguments.
        
        Args:
            *args: Arguments to pass to the C/C++ function
        
        Returns:
            Return value from the C/C++ function
        """
        if not self.function_handle:
            raise CRuntimeError("Function handle not available")
        
        try:
            return self.function_handle(*args)
        except Exception as e:
            raise CRuntimeError(f"Failed to execute C/C++ function: {str(e)}")
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the kernel.
        
        This method should be called when the kernel is no longer needed
        to ensure proper resource cleanup.
        """
        # Release the library handle
        if self.lib_handle:
            if sys.platform == 'win32':
                # Windows-specific cleanup
                ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                ctypes.windll.kernel32.FreeLibrary(self.lib_handle._handle)
            self.lib_handle = None
            self.function_handle = None
        
        # Delete the library file if we own it
        if self.own_lib:
            try:
                if os.path.exists(self.lib_path):
                    os.unlink(self.lib_path)
            except (OSError, IOError) as e:
                logger.warning(f"Failed to delete library file '{self.lib_path}': {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class CBackend:
    """
    Backend for executing C/C++ code.
    
    This class provides a high-level interface for generating, compiling,
    and executing architecture-specific C/C++ code.
    """
    
    def __init__(self, 
                 arch: str = None, 
                 compiler: str = None,
                 cpp: bool = True,
                 optimization_level: str = "-O3"):
        """
        Initialize the C/C++ backend.
        
        Args:
            arch: Target architecture (auto-detected if None)
            compiler: Compiler to use (auto-detected if None)
            cpp: Whether to use C++ (True) or C (False)
            optimization_level: Optimization level for the compiler
        """
        self.arch = arch or self._detect_architecture()
        self.cpp = cpp
        self.optimization_level = optimization_level
        
        # Initialize code generator and compiler
        self.code_generator = CCodeGenerator(arch=self.arch, cpp=self.cpp)
        self.compiler = CompilerWrapper(
            compiler=compiler,
            arch=self.arch,
            optimization_level=self.optimization_level,
            cpp=self.cpp
        )
        
        # Cache for compiled kernels (function_name -> CompiledKernel)
        self.kernels = {}
    
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
            raise CRuntimeError(f"Unsupported architecture: {machine}")
    
    def compile_and_load(self, 
                        source_code: str, 
                        function_name: str, 
                        arg_types: List[type], 
                        return_type: type = None,
                        output_file: Optional[str] = None) -> CompiledKernel:
        """
        Compile and load C/C++ code as a callable kernel.
        
        Args:
            source_code: C/C++ source code to compile
            function_name: Name of the function in the C/C++ code
            arg_types: List of ctypes argument types
            return_type: Return type of the function (None for void)
            output_file: Optional path for the output library
        
        Returns:
            CompiledKernel instance
        """
        try:
            # Compile the code
            lib_path = self.compiler.compile(source_code, output_file=output_file)
            
            # Create and return the kernel
            return CompiledKernel(
                lib_path=lib_path,
                function_name=function_name,
                arg_types=arg_types,
                return_type=return_type,
                own_lib=(output_file is None)  # Own the lib if we created a temporary file
            )
            
        except CompilerError as e:
            raise CRuntimeError(f"Failed to compile C/C++ code: {str(e)}")
    
    def create_vector_add_kernel(self, 
                               function_name: str = "vector_add",
                               data_type: str = "float") -> CompiledKernel:
        """
        Create a kernel for vector addition.
        
        Args:
            function_name: Name of the function
            data_type: Data type for the vectors
        
        Returns:
            CompiledKernel for vector addition
        """
        # Generate source code
        source_code = self.code_generator.generate_full_source(
            function_name=function_name,
            operation="vector_add",
            data_type=data_type
        )
        
        # Determine argument types based on data_type
        if data_type == "float":
            c_type = ctypes.c_float
        elif data_type == "double":
            c_type = ctypes.c_double
        elif data_type == "int":
            c_type = ctypes.c_int
        else:
            raise CRuntimeError(f"Unsupported data type: {data_type}")
        
        # Compile and load
        return self.compile_and_load(
            source_code=source_code,
            function_name=function_name,
            arg_types=[
                ctypes.POINTER(c_type),  # Input array 1
                ctypes.POINTER(c_type),  # Input array 2
                ctypes.POINTER(c_type),  # Output array
                ctypes.c_int             # Size
            ],
            return_type=None  # void function
        )
    
    def create_matrix_multiply_kernel(self, 
                                     function_name: str = "matrix_multiply",
                                     data_type: str = "float") -> CompiledKernel:
        """
        Create a kernel for matrix multiplication.
        
        Args:
            function_name: Name of the function
            data_type: Data type for the matrices
        
        Returns:
            CompiledKernel for matrix multiplication
        """
        # Generate source code
        source_code = self.code_generator.generate_full_source(
            function_name=function_name,
            operation="matrix_multiply",
            data_type=data_type
        )
        
        # Determine argument types based on data_type
        if data_type == "float":
            c_type = ctypes.c_float
        elif data_type == "double":
            c_type = ctypes.c_double
        elif data_type == "int":
            c_type = ctypes.c_int
        else:
            raise CRuntimeError(f"Unsupported data type: {data_type}")
        
        # Compile and load
        return self.compile_and_load(
            source_code=source_code,
            function_name=function_name,
            arg_types=[
                ctypes.POINTER(c_type),  # Input matrix A
                ctypes.POINTER(c_type),  # Input matrix B
                ctypes.POINTER(c_type),  # Output matrix C
                ctypes.c_int,            # m (rows of A and C)
                ctypes.c_int,            # n (columns of B and C)
                ctypes.c_int             # k (columns of A and rows of B)
            ],
            return_type=None  # void function
        )
    
    def vector_add(self, a, b, data_type: str = "float"):
        """
        Perform vector addition using optimized C/C++ code.
        
        Args:
            a: First input array
            b: Second input array
            data_type: Data type of the arrays
        
        Returns:
            Result array
        """
        import numpy as np
        
        # Ensure inputs are numpy arrays of the right type
        if data_type == "float":
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            c_type = ctypes.c_float
        elif data_type == "double":
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            c_type = ctypes.c_double
        elif data_type == "int":
            a = np.asarray(a, dtype=np.int32)
            b = np.asarray(b, dtype=np.int32)
            c_type = ctypes.c_int
        else:
            raise CRuntimeError(f"Unsupported data type: {data_type}")
        
        # Check shapes
        if a.shape != b.shape:
            raise ValueError(f"Input arrays must have the same shape: {a.shape} vs {b.shape}")
        
        # Create output array
        c = np.zeros_like(a)
        
        # Get or create the kernel
        kernel_key = f"vector_add_{data_type}"
        if kernel_key not in self.kernels:
            self.kernels[kernel_key] = self.create_vector_add_kernel(
                function_name="vector_add",
                data_type=data_type
            )
        
        kernel = self.kernels[kernel_key]
        
        # Create contiguous arrays and get ctypes pointers
        a_flat = np.ascontiguousarray(a.flatten())
        b_flat = np.ascontiguousarray(b.flatten())
        c_flat = np.ascontiguousarray(c.flatten())
        
        a_ptr = a_flat.ctypes.data_as(ctypes.POINTER(c_type))
        b_ptr = b_flat.ctypes.data_as(ctypes.POINTER(c_type))
        c_ptr = c_flat.ctypes.data_as(ctypes.POINTER(c_type))
        
        # Call the kernel
        size = np.prod(a.shape)
        kernel(a_ptr, b_ptr, c_ptr, ctypes.c_int(size))
        
        # Reshape the result to match the input shape
        return c_flat.reshape(a.shape)
    
    def matrix_multiply(self, a, b, data_type: str = "float"):
        """
        Perform matrix multiplication using optimized C/C++ code.
        
        Args:
            a: First input matrix
            b: Second input matrix
            data_type: Data type of the matrices
        
        Returns:
            Result matrix
        """
        import numpy as np
        
        # Ensure inputs are numpy arrays of the right type
        if data_type == "float":
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            c_type = ctypes.c_float
        elif data_type == "double":
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            c_type = ctypes.c_double
        elif data_type == "int":
            a = np.asarray(a, dtype=np.int32)
            b = np.asarray(b, dtype=np.int32)
            c_type = ctypes.c_int
        else:
            raise CRuntimeError(f"Unsupported data type: {data_type}")
        
        # Check shapes
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Input matrices must be 2D: {a.shape} and {b.shape}")
        
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Matrix shapes incompatible for multiplication: {a.shape} and {b.shape}")
        
        # Extract dimensions
        m, k = a.shape
        k2, n = b.shape
        
        # Create output matrix
        c = np.zeros((m, n), dtype=a.dtype)
        
        # Get or create the kernel
        kernel_key = f"matrix_multiply_{data_type}"
        if kernel_key not in self.kernels:
            self.kernels[kernel_key] = self.create_matrix_multiply_kernel(
                function_name="matrix_multiply",
                data_type=data_type
            )
        
        kernel = self.kernels[kernel_key]
        
        # Create contiguous arrays and get ctypes pointers
        a_cont = np.ascontiguousarray(a)
        b_cont = np.ascontiguousarray(b)
        c_cont = np.ascontiguousarray(c)
        
        a_ptr = a_cont.ctypes.data_as(ctypes.POINTER(c_type))
        b_ptr = b_cont.ctypes.data_as(ctypes.POINTER(c_type))
        c_ptr = c_cont.ctypes.data_as(ctypes.POINTER(c_type))
        
        # Call the kernel
        kernel(a_ptr, b_ptr, c_ptr, 
              ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k))
        
        return c_cont
    
    def cleanup(self):
        """
        Clean up resources used by the backend.
        
        This method should be called when the backend is no longer needed
        to ensure proper resource cleanup.
        """
        # Clean up all cached kernels
        for kernel in self.kernels.values():
            kernel.cleanup()
        self.kernels = {}