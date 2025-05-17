"""
Prism hardware-specific backends for different compute architectures.

This package provides backend implementations for various hardware accelerators,
with specialized optimizations for specific hardware versions, as well as
low-level backend implementations (Assembly, C/C++) for high-performance computing.
"""

import os
import sys
import logging
import platform
import importlib
from typing import Dict, List, Any, Optional, Union, Callable

# Device-specific backends
try:
    from .trainium import TrainiumDevice
    from .trainium_v1 import Trainium1Device
    from .trainium_v2 import Trainium2Device
    from .inferentia import InferentiaDevice
    from .inferentia_v1 import Inferentia1Device
    from .inferentia_v2 import Inferentia2Device
    from .graviton import (
        GravitonDevice, 
        Graviton1Device, 
        Graviton2Device, 
        Graviton3Device,
        Graviton3EDevice,
        Graviton4Device
    )
    from .fpga import FPGADevice
    from .apple_silicon import (
        AppleSiliconDevice,
        M1Device, M1ProDevice, M1MaxDevice, M1UltraDevice,
        M2Device, M2ProDevice, M2MaxDevice, M2UltraDevice,
        M3Device, M3ProDevice, M3MaxDevice, M3UltraDevice,
        M4Device, M4ProDevice, M4MaxDevice
    )
    
    # Map of backend names to device classes
    DEVICE_BACKENDS = {
        # AWS Silicon
        "trainium": TrainiumDevice,
        "trainium1": Trainium1Device,
        "trainium2": Trainium2Device,
        "inferentia": InferentiaDevice,
        "inferentia1": Inferentia1Device,
        "inferentia2": Inferentia2Device,
        "graviton": GravitonDevice,
        "graviton1": Graviton1Device,
        "graviton2": Graviton2Device,
        "graviton3": Graviton3Device,
        "graviton3e": Graviton3EDevice,
        "graviton4": Graviton4Device,
        "fpga": FPGADevice,
        
        # Apple Silicon
        "apple_silicon": AppleSiliconDevice,
        "m1": M1Device,
        "m1pro": M1ProDevice,
        "m1max": M1MaxDevice,
        "m1ultra": M1UltraDevice,
        "m2": M2Device,
        "m2pro": M2ProDevice,
        "m2max": M2MaxDevice,
        "m2ultra": M2UltraDevice,
        "m3": M3Device,
        "m3pro": M3ProDevice,
        "m3max": M3MaxDevice,
        "m3ultra": M3UltraDevice,
        "m4": M4Device,
        "m4pro": M4ProDevice,
        "m4max": M4MaxDevice,
    }
except ImportError:
    # If device-specific backends aren't available, create an empty dict
    DEVICE_BACKENDS = {}

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Registry for Prism backends.
    
    This class maintains a registry of available backends with their capabilities,
    allowing the framework to discover and select appropriate backends at runtime.
    """
    
    def __init__(self):
        """Initialize the backend registry."""
        self.backends = {}
        self.backend_capabilities = {}
    
    def register_backend(self, 
                         name: str, 
                         backend_class: Any, 
                         capabilities: Dict[str, Any] = None):
        """
        Register a backend with the registry.
        
        Args:
            name: Name of the backend
            backend_class: Backend class or factory function
            capabilities: Dictionary of backend capabilities
        """
        self.backends[name] = backend_class
        self.backend_capabilities[name] = capabilities or {}
        
        logger.info(f"Registered backend: {name}")
    
    def get_backend(self, name: str) -> Any:
        """
        Get a backend by name.
        
        Args:
            name: Name of the backend
        
        Returns:
            Backend instance
        
        Raises:
            ValueError: If the backend is not found
        """
        if name not in self.backends:
            raise ValueError(f"Backend not found: {name}")
        
        # If the backend is a class, instantiate it
        backend = self.backends[name]
        if isinstance(backend, type):
            return backend()
        
        # If it's a factory function, call it
        if callable(backend):
            return backend()
        
        # Otherwise, assume it's already instantiated
        return backend
    
    def get_capabilities(self, name: str) -> Dict[str, Any]:
        """
        Get capabilities of a backend.
        
        Args:
            name: Name of the backend
        
        Returns:
            Dictionary of backend capabilities
        
        Raises:
            ValueError: If the backend is not found
        """
        if name not in self.backend_capabilities:
            raise ValueError(f"Backend not found: {name}")
        
        return self.backend_capabilities[name]
    
    def find_backends_for_operation(self, operation: str) -> List[str]:
        """
        Find backends that support a specific operation.
        
        Args:
            operation: Name of the operation
        
        Returns:
            List of backend names that support the operation
        """
        return [
            name for name, capabilities in self.backend_capabilities.items()
            if operation in capabilities.get('operations', [])
        ]
    
    def find_backends_for_arch(self, arch: str) -> List[str]:
        """
        Find backends optimized for a specific architecture.
        
        Args:
            arch: Architecture identifier
        
        Returns:
            List of backend names optimized for the architecture
        """
        return [
            name for name, capabilities in self.backend_capabilities.items()
            if arch in capabilities.get('architectures', [])
        ]
    
    def list_backends(self) -> List[str]:
        """
        List all registered backends.
        
        Returns:
            List of backend names
        """
        return list(self.backends.keys())


# Global registry instance
_registry = BackendRegistry()


def get_backend_registry() -> BackendRegistry:
    """
    Get the global backend registry.
    
    Returns:
        Global backend registry instance
    """
    return _registry


# Auto-register available backends
def _register_builtin_backends():
    """Register built-in backends with the registry."""
    # Register the C/C++ backend if available
    try:
        from .cbackend import CBackend
        
        _registry.register_backend(
            'c', 
            CBackend,
            capabilities={
                'operations': ['vector_add', 'matrix_multiply'],
                'architectures': ['x86_64', 'arm64'],
                'description': 'C/C++ backend with architecture-specific optimizations',
                'features': ['simd', 'compiler_optimizations']
            }
        )
    except ImportError:
        logger.debug("C/C++ backend not available")
    
    # Register the Assembly backend if available
    try:
        from .asmbackend import AssemblyBackend
        
        _registry.register_backend(
            'assembly',
            AssemblyBackend,
            capabilities={
                'operations': ['vector_add', 'matrix_multiply'],
                'architectures': ['x86_64', 'arm64'],
                'description': 'Assembly backend with architecture-specific optimizations',
                'features': ['simd', 'direct_assembly']
            }
        )
    except ImportError:
        logger.debug("Assembly backend not available")
    
    # Register device-specific backends from DEVICE_BACKENDS
    for name, backend_class in DEVICE_BACKENDS.items():
        arch = 'arm64' if 'graviton' in name or 'apple' in name or 'm' in name else 'custom'
        
        capabilities = {
            'operations': ['forward', 'backward'],
            'architectures': [arch],
            'description': f'Hardware-specific backend for {name}',
            'features': ['hardware_acceleration']
        }
        
        # Add specific features based on backend name
        if 'graviton' in name:
            capabilities['features'].extend(['neon', 'sve'])
        elif 'apple' in name or any(m in name for m in ['m1', 'm2', 'm3', 'm4']):
            capabilities['features'].extend(['neural_engine', 'amx'])
        elif 'trainium' in name:
            capabilities['features'].extend(['training_acceleration'])
        elif 'inferentia' in name:
            capabilities['features'].extend(['inference_acceleration'])
        
        _registry.register_backend(name, backend_class, capabilities)


# Register built-in backends
_register_builtin_backends()


# Function to get appropriate device backend class (legacy API)
def get_backend(name, **kwargs):
    """
    Get backend implementation by name.
    
    Args:
        name: Backend name
        **kwargs: Additional parameters for backend initialization
        
    Returns:
        Backend device instance
        
    Raises:
        ValueError: If backend is not supported
    """
    # First check the registry
    try:
        return _registry.get_backend(name)
    except ValueError:
        pass
    
    # Then check device backends (legacy)
    if name not in DEVICE_BACKENDS:
        raise ValueError(f"Unsupported backend: {name}")
    
    return DEVICE_BACKENDS[name](**kwargs)