"""
C/C++ backend for the Prism framework.

This module provides a backend that generates, compiles, and executes
optimized C/C++ code for various processor architectures.
"""

from .codegen import CCodeGenerator
from .compiler import CompilerWrapper
from .runtime import CompiledKernel, CBackend
from .templates import get_template

__all__ = [
    'CCodeGenerator',
    'CompilerWrapper',
    'CompiledKernel',
    'CBackend',
    'get_template',
]