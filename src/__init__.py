"""
Prism: Parallel Research Infrastructure for Scientific Multiprocessing

A framework for scientific computing on AWS Silicon and other hardware platforms.
"""

from .core import __version__

# Import device-related functionality
from .core.device import Device, get_device, get_available_devices, DeviceContext
from .core.capability import CapabilityDetector, BackendSelector

# Import low-level operations
from .core.lowlevel import vector_add, matrix_multiply

# Import from .core
from .core import detector, selector

# Global state for device management
_active_device = None
_active_backend = "auto"

def get_active_device() -> Device:
    """
    Get the currently active device.
    
    Returns:
        The currently active device
    
    Raises:
        RuntimeError: If no device is active
    """
    global _active_device
    
    if _active_device is None:
        # Initialize with default device on first call
        _active_device = get_device()
        
    return _active_device

def set_active_device(device: Device) -> None:
    """
    Set the active device.
    
    Args:
        device: Device to set as active
    """
    global _active_device
    _active_device = device

def get_current_backend() -> str:
    """
    Get the current backend.
    
    Returns:
        Current backend name
    """
    global _active_backend
    return _active_backend

def set_current_backend(backend_name: str) -> None:
    """
    Set the current backend.
    
    Args:
        backend_name: Name of the backend to use
    """
    global _active_backend
    _active_backend = backend_name

def device(device_type: str = None, device_id: int = 0, version=None):
    """
    Context manager for temporarily switching the active device.
    
    This is a convenience wrapper around DeviceContext that allows for a cleaner API:
    ```
    with prism.device("aws.graviton", version="3E"):
        # Code executed with Graviton 3E active
    ```
    
    Args:
        device_type: Device type to use (e.g., "aws.graviton", "aws.trainium")
        device_id: Device ID
        version: Optional version specification (e.g., 1, 2, 3, "3E", 4 for Graviton)
        
    Returns:
        DeviceContext instance that acts as a context manager
    """
    return DeviceContext(device_type, device_id, version)


def use_backend(backend_name: str):
    """
    Context manager for temporarily switching the active backend.
    
    Example:
    ```
    with prism.use_backend("assembly"):
        # Code executed with assembly backend
    ```
    
    Args:
        backend_name: Name of the backend to use ("assembly", "c", or "auto")
        
    Returns:
        Context manager for backend selection
    """
    class BackendContext:
        def __init__(self, backend):
            self.backend = backend
            self.previous_backend = None
            
        def __enter__(self):
            self.previous_backend = get_current_backend()
            set_current_backend(self.backend)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            set_current_backend(self.previous_backend)
            
    return BackendContext(backend_name)


def use_assembly_backend():
    """
    Context manager for assembly backend selection.
    
    Returns:
        Context manager for assembly backend
    """
    return use_backend("assembly")


def use_c_backend():
    """
    Context manager for C/C++ backend selection.
    
    Returns:
        Context manager for C/C++ backend
    """
    return use_backend("c")

# Specialized device selectors for AWS Graviton
def graviton(version=None, device_id: int = 0):
    """
    Context manager for temporarily switching to a Graviton device.
    
    Example:
    ```
    with prism.graviton(version=3):
        # Code executed with Graviton 3 active
    ```
    
    Args:
        version: Graviton version (1, 2, 3, "3E", or 4)
        device_id: Device ID (usually 0 for Graviton)
        
    Returns:
        DeviceContext instance that acts as a context manager
    """
    return DeviceContext("aws.graviton", device_id, version)

# Specialized Graviton version selectors
def graviton1(device_id: int = 0):
    """Context manager for Graviton 1 device."""
    return graviton(version=1, device_id=device_id)

def graviton2(device_id: int = 0):
    """Context manager for Graviton 2 device."""
    return graviton(version=2, device_id=device_id)

def graviton3(device_id: int = 0):
    """Context manager for Graviton 3 device."""
    return graviton(version=3, device_id=device_id)

def graviton3e(device_id: int = 0):
    """Context manager for Graviton 3E device."""
    return graviton(version="3E", device_id=device_id)

def graviton4(device_id: int = 0):
    """Context manager for Graviton 4 device."""
    return graviton(version=4, device_id=device_id)

# Specialized device selectors for AWS Trainium/Inferentia
def trainium(version=None, device_id: int = 0):
    """Context manager for Trainium device."""
    return DeviceContext("aws.trainium", device_id, version)

def inferentia(version=None, device_id: int = 0):
    """Context manager for Inferentia device."""
    return DeviceContext("aws.inferentia", device_id, version)

# Specialized device selectors for Apple Silicon
def apple_silicon(version=None, device_id: int = 0):
    """
    Context manager for temporarily switching to an Apple Silicon device.
    
    Example:
    ```
    with prism.apple_silicon(version="M1Pro"):
        # Code executed with M1 Pro active
    ```
    
    Args:
        version: Apple Silicon version (M1, M1Pro, M1Max, M1Ultra, M2, etc.)
        device_id: Device ID (usually 0 for Apple Silicon)
        
    Returns:
        DeviceContext instance that acts as a context manager
    """
    return DeviceContext("apple.silicon", device_id, version)

# Specialized Apple Silicon selectors - M1 series
def m1(device_id: int = 0):
    """Context manager for Apple M1 device."""
    return apple_silicon(version="M1", device_id=device_id)

def m1_pro(device_id: int = 0):
    """Context manager for Apple M1 Pro device."""
    return apple_silicon(version="M1Pro", device_id=device_id)

def m1_max(device_id: int = 0):
    """Context manager for Apple M1 Max device."""
    return apple_silicon(version="M1Max", device_id=device_id)

def m1_ultra(device_id: int = 0):
    """Context manager for Apple M1 Ultra device."""
    return apple_silicon(version="M1Ultra", device_id=device_id)

# Specialized Apple Silicon selectors - M2 series
def m2(device_id: int = 0):
    """Context manager for Apple M2 device."""
    return apple_silicon(version="M2", device_id=device_id)

def m2_pro(device_id: int = 0):
    """Context manager for Apple M2 Pro device."""
    return apple_silicon(version="M2Pro", device_id=device_id)

def m2_max(device_id: int = 0):
    """Context manager for Apple M2 Max device."""
    return apple_silicon(version="M2Max", device_id=device_id)

def m2_ultra(device_id: int = 0):
    """Context manager for Apple M2 Ultra device."""
    return apple_silicon(version="M2Ultra", device_id=device_id)

# Specialized Apple Silicon selectors - M3 series
def m3(device_id: int = 0):
    """Context manager for Apple M3 device."""
    return apple_silicon(version="M3", device_id=device_id)

def m3_pro(device_id: int = 0):
    """Context manager for Apple M3 Pro device."""
    return apple_silicon(version="M3Pro", device_id=device_id)

def m3_max(device_id: int = 0):
    """Context manager for Apple M3 Max device."""
    return apple_silicon(version="M3Max", device_id=device_id)

def m3_ultra(device_id: int = 0):
    """Context manager for Apple M3 Ultra device."""
    return apple_silicon(version="M3Ultra", device_id=device_id)

# Specialized Apple Silicon selectors - M4 series
def m4(device_id: int = 0):
    """Context manager for Apple M4 device."""
    return apple_silicon(version="M4", device_id=device_id)

def m4_pro(device_id: int = 0):
    """Context manager for Apple M4 Pro device."""
    return apple_silicon(version="M4Pro", device_id=device_id)

def m4_max(device_id: int = 0):
    """Context manager for Apple M4 Max device."""
    return apple_silicon(version="M4Max", device_id=device_id)

# Function to create optimal device based on workload
def create_optimal_device(workload_type="inference", **kwargs):
    """
    Create optimal device for the specified workload.
    
    Args:
        workload_type: Type of workload (training, inference, hpc, etc.)
        **kwargs: Additional workload characteristics
        
    Returns:
        Optimal device instance for the workload
    """
    from .core import selector
    return selector.create_optimal_device(workload_type, **kwargs)