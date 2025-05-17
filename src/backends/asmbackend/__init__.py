"""
Assembly backend for the Prism framework.

This module provides backends for generating, assembling, and executing
architecture-specific assembly code for ARM and x86_64 processors.
"""

from .armasm import ARMAssemblyGenerator
from .x86asm import X86AssemblyGenerator
from .assembler import Assembler, ARMAssembler, X86Assembler, get_assembler
from .runtime import AssemblyKernel, AssemblyBackend
from .examples import ARMExamples, X86Examples, get_examples

__all__ = [
    'ARMAssemblyGenerator',
    'X86AssemblyGenerator',
    'Assembler',
    'ARMAssembler',
    'X86Assembler',
    'get_assembler',
    'AssemblyKernel',
    'AssemblyBackend',
    'ARMExamples',
    'X86Examples',
    'get_examples',
]