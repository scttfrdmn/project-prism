"""
Low-level backend interface.

This module defines the interface for low-level backends (C/C++ and assembly)
that can be used to generate optimized code for specific architectures.
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, Union, List

from .operation import OperationGraph

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Enumeration of supported backend types."""
    C = 'c'
    CPP = 'cpp'
    ASSEMBLY = 'assembly'


class Architecture(Enum):
    """Enumeration of supported architectures."""
    ARM = 'arm'
    AMD64 = 'amd64'
    INTEL64 = 'intel64'
    GENERIC = 'generic'


class LowLevelBackend:
    """Interface for low-level backends."""
    
    def __init__(self, 
                 backend_type: Union[BackendType, str], 
                 architecture: Union[Architecture, str], 
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize low-level backend.
        
        Args:
            backend_type: Backend type (c, cpp, assembly)
            architecture: Target architecture (arm, amd64, intel64)
            options: Additional backend options
        """
        # Convert string to enum if needed
        if isinstance(backend_type, str):
            backend_type = BackendType(backend_type)
        if isinstance(architecture, str):
            try:
                architecture = Architecture(architecture)
            except ValueError:
                architecture = Architecture.GENERIC
                logger.warning(f"Unknown architecture: {architecture}, using generic")
        
        self.backend_type = backend_type
        self.architecture = architecture
        self.options = options or {}
        
        # Detect available instruction sets
        self.available_instruction_sets = self._detect_instruction_sets()
        
        # Filter requested instruction sets to what's available
        self.instruction_sets = self._filter_instruction_sets(
            self.options.get('instruction_sets', [])
        )
        
        logger.info(f"Initialized {backend_type.value} backend for {architecture.value} "
                    f"with instruction sets: {self.instruction_sets}")
    
    def _detect_instruction_sets(self) -> List[str]:
        """
        Detect available instruction sets for target architecture.
        
        Returns:
            List of available instruction set names
        """
        # This is a placeholder implementation that should be overridden by subclasses
        if self.architecture == Architecture.ARM:
            return ["neon", "sve", "sve2"]
        elif self.architecture in (Architecture.AMD64, Architecture.INTEL64):
            return ["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx", "avx2", "avx512f"]
        else:
            return []
    
    def _filter_instruction_sets(self, requested_sets: List[str]) -> List[str]:
        """
        Filter requested instruction sets to what's available.
        
        Args:
            requested_sets: List of requested instruction set names
            
        Returns:
            List of available requested instruction set names
        """
        if not requested_sets:
            # Use all available instruction sets if none explicitly requested
            return self.available_instruction_sets
        
        # Filter to what's available
        filtered_sets = [
            iset for iset in requested_sets if iset in self.available_instruction_sets
        ]
        
        if not filtered_sets and requested_sets:
            logger.warning(f"None of the requested instruction sets {requested_sets} are available. "
                           f"Available sets: {self.available_instruction_sets}")
        
        return filtered_sets
    
    def compile(self, 
                operation_graph: OperationGraph, 
                function_name: str, 
                output_path: Optional[str] = None) -> Any:
        """
        Compile operation graph into executable.
        
        Args:
            operation_graph: Computational operation graph
            function_name: Name of the entry point function
            output_path: Optional path for output files
            
        Returns:
            Compiled kernel object
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement compile")
    
    def is_available(self) -> bool:
        """
        Check if this backend is available on the current system.
        
        Returns:
            True if available, False otherwise
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement is_available")