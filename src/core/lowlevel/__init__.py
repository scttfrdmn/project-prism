"""
Low-level backend integration for the Prism framework.

This module provides interfaces and infrastructure for integrating low-level
backends (C/C++, Assembly) with the core Prism framework, enabling highly 
optimized architecture-specific implementations of operations.
"""

from .backend import LowLevelBackend
from .operation import Operation, OperationGraph
from .integration import (
    registry,
    get_optimal_backend_for_operation,
    execute_operation,
    vector_add,
    matrix_multiply,
)

__all__ = [
    'LowLevelBackend',
    'Operation',
    'OperationGraph',
    'registry',
    'get_optimal_backend_for_operation',
    'execute_operation',
    'vector_add',
    'matrix_multiply',
]