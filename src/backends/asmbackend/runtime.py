"""
Runtime execution for assembly code in the Prism framework.

This module provides classes for loading and executing compiled assembly
code as dynamically loaded libraries using ctypes.
"""

import os
import sys
import ctypes
import platform
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from .assembler import get_assembler, AssemblerError


class AssemblyRuntimeError(Exception):
    """Exception raised for errors during assembly code execution."""
    pass


class AssemblyKernel:
    """
    Represents a compiled assembly kernel that can be executed.
    
    This class handles loading the compiled assembly code as a shared library
    and providing a Python callable interface to the assembly functions.
    """
    
    def __init__(self, 
                 object_file: str, 
                 function_name: str, 
                 arg_types: List[type], 
                 return_type: type = None):
        """
        Initialize the assembly kernel.
        
        Args:
            object_file: Path to the compiled object file
            function_name: Name of the function in the assembly code
            arg_types: List of ctypes argument types
            return_type: Return type of the function (None for void)
        """
        self.object_file = object_file
        self.function_name = function_name
        self.arg_types = arg_types
        self.return_type = return_type
        self.lib_handle = None
        self.function_handle = None
        
        # Create the shared library from the object file
        self.shared_lib_path = self._create_shared_library()
        
        # Load the shared library
        self._load_library()
    
    def _create_shared_library(self) -> str:
        """
        Create a shared library from the object file.
        
        Returns:
            Path to the created shared library
        """
        # Determine library file extension
        if sys.platform == 'darwin':
            lib_extension = '.dylib'
        elif sys.platform == 'win32':
            lib_extension = '.dll'
        else:
            lib_extension = '.so'
        
        # Create output path
        lib_path = os.path.splitext(self.object_file)[0] + lib_extension
        
        # Build the link command
        if sys.platform == 'darwin':
            cmd = [
                'clang',
                '-shared',
                '-o', lib_path,
                self.object_file
            ]
        elif sys.platform == 'win32':
            cmd = [
                'link',
                '/DLL',
                '/OUT:' + lib_path,
                self.object_file
            ]
        else:
            cmd = [
                'gcc',
                '-shared',
                '-o', lib_path,
                self.object_file
            ]
        
        # Run the link command
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise AssemblyRuntimeError(f"Failed to create shared library: {result.stderr}")
            
            return lib_path
        except subprocess.SubprocessError as e:
            raise AssemblyRuntimeError(f"Failed to run linker: {str(e)}")
    
    def _load_library(self) -> None:
        """Load the shared library and set up the function handle."""
        try:
            # Load the shared library
            self.lib_handle = ctypes.CDLL(self.shared_lib_path)
            
            # Get the function from the library
            func = getattr(self.lib_handle, self.function_name)
            
            # Set the function argument and return types
            func.argtypes = self.arg_types
            func.restype = self.return_type
            
            # Store the function handle
            self.function_handle = func
        except (AttributeError, OSError) as e:
            raise AssemblyRuntimeError(f"Failed to load the shared library: {str(e)}")
    
    def __call__(self, *args):
        """
        Call the assembly function with the given arguments.
        
        Args:
            *args: Arguments to pass to the assembly function
        
        Returns:
            Return value from the assembly function
        """
        if not self.function_handle:
            raise AssemblyRuntimeError("Function handle not available")
        
        try:
            return self.function_handle(*args)
        except Exception as e:
            raise AssemblyRuntimeError(f"Failed to execute assembly function: {str(e)}")
    
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
        
        # Delete the files if they exist
        for file_path in [self.object_file, self.shared_lib_path]:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except (OSError, IOError):
                # Non-critical error, just continue
                pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class AssemblyBackend:
    """
    Backend for executing assembly code.
    
    This class provides a high-level interface for generating, assembling,
    and executing architecture-specific assembly code.
    """
    
    def __init__(self, arch: str = None, syntax: str = "att"):
        """
        Initialize the assembly backend.
        
        Args:
            arch: Target architecture (auto-detected if None)
            syntax: Assembly syntax style ("att" or "intel" for x86_64)
        """
        self.arch = arch or self._detect_architecture()
        self.syntax = syntax
        self.assembler = get_assembler(self.arch, syntax)
        self.kernels = {}  # Cache for compiled kernels
    
    def _detect_architecture(self) -> str:
        """
        Detect the current system architecture.
        
        Returns:
            Architecture identifier string
        """
        machine = platform.machine().lower()
        if machine in ("arm", "aarch64", "arm64"):
            return "arm64"
        elif machine in ("x86_64", "amd64", "x64"):
            return "x86_64"
        else:
            raise AssemblyRuntimeError(f"Unsupported architecture: {machine}")
    
    def compile_and_load(self, 
                         asm_code: str, 
                         function_name: str, 
                         arg_types: List[type], 
                         return_type: type = None) -> AssemblyKernel:
        """
        Compile and load assembly code as a callable kernel.
        
        Args:
            asm_code: Assembly code to compile
            function_name: Name of the function in the assembly code
            arg_types: List of ctypes argument types
            return_type: Return type of the function (None for void)
        
        Returns:
            AssemblyKernel instance
        """
        # Assemble the code
        try:
            object_file = self.assembler.assemble(asm_code)
        except AssemblerError as e:
            raise AssemblyRuntimeError(f"Failed to assemble code: {str(e)}")
        
        # Create and return the kernel
        return AssemblyKernel(object_file, function_name, arg_types, return_type)
    
    def execute(self, 
                asm_code: str, 
                function_name: str, 
                args: List[Any], 
                arg_types: List[type], 
                return_type: type = None,
                cache: bool = True) -> Any:
        """
        Compile and execute assembly code.
        
        Args:
            asm_code: Assembly code to compile and execute
            function_name: Name of the function in the assembly code
            args: List of arguments to pass to the function
            arg_types: List of ctypes argument types
            return_type: Return type of the function (None for void)
            cache: Whether to cache the compiled kernel
        
        Returns:
            Return value from the assembly function
        """
        # Check if we have a cached kernel
        cache_key = hash(asm_code + function_name + str(arg_types) + str(return_type))
        kernel = self.kernels.get(cache_key) if cache else None
        
        if not kernel:
            # Compile and load the kernel
            kernel = self.compile_and_load(asm_code, function_name, arg_types, return_type)
            
            # Cache the kernel if requested
            if cache:
                self.kernels[cache_key] = kernel
        
        # Execute the kernel
        return kernel(*args)
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the backend.
        
        This method should be called when the backend is no longer needed
        to ensure proper resource cleanup.
        """
        # Clean up all cached kernels
        for kernel in self.kernels.values():
            kernel.cleanup()
        self.kernels = {}