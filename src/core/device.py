"""
Device abstractions for hardware management.
"""

from typing import List, Optional, Dict, Any, Tuple, Union


class Device:
    """
    Abstract representation of a compute device.
    """
    
    def __init__(self, device_id: Any = None, device_type: Optional[str] = None):
        """
        Initialize a device instance.
        
        Args:
            device_id: Optional device identifier
            device_type: Optional device type (e.g., "aws.graviton", "aws.trainium", "nvidia.h100")
        """
        self.device_id = device_id
        self.device_type = device_type
        self._capabilities = self._get_capabilities()
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """
        Get device capabilities.
        
        Returns:
            Dictionary of device capabilities
        """
        return {}
    
    def allocate_memory(self, size_bytes: int) -> Any:
        """
        Allocate device memory.
        
        Args:
            size_bytes: Size in bytes to allocate
            
        Returns:
            Memory handle
        """
        raise NotImplementedError("Subclasses must implement allocate_memory")
    
    def copy_to_device(self, host_data: Any, device_data: Any, size_bytes: int) -> None:
        """
        Copy data from host to device.
        
        Args:
            host_data: Host data source
            device_data: Device data destination
            size_bytes: Size in bytes to copy
        """
        raise NotImplementedError("Subclasses must implement copy_to_device")
    
    def synchronize(self) -> None:
        """
        Synchronize device operations.
        """
        raise NotImplementedError("Subclasses must implement synchronize")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get device capabilities.
        
        Returns:
            Dictionary of device capabilities
        """
        # Default implementation returns capabilities from initialization
        return self._capabilities


def get_device(device_type: Optional[str] = None, device_id: Optional[int] = None, 
               version: Optional[Union[int, str]] = None) -> Device:
    """
    Get a device instance of the specified type.
    
    Args:
        device_type: Device type to get (e.g., "aws.graviton", "aws.trainium")
        device_id: Optional device ID
        version: Optional specific version of the device
        
    Returns:
        Device instance
    
    Raises:
        ValueError: If the requested device type is not available
        ImportError: If module for device type cannot be imported
    """
    # Import here to avoid circular imports
    from ..backends import get_backend
    
    # Parse device type
    if device_type is None:
        # Use default device from capability detector
        from .capability import detector
        return detector.create_device()
    
    # Parse AWS device types
    if device_type.startswith("aws."):
        backend_name = device_type[4:]  # Remove "aws." prefix
        
        # Handle version-specific backends
        if version is not None:
            if backend_name == "graviton":
                if version in (1, 2, 3, 4) or version == "3E":
                    backend_name = f"graviton{version}"
            elif backend_name == "trainium" and version in (1, 2):
                backend_name = f"trainium{version}"
            elif backend_name == "inferentia" and version in (1, 2):
                backend_name = f"inferentia{version}"
        
        try:
            return get_backend(backend_name, device_id=device_id)
        except (ValueError, ImportError) as e:
            raise ValueError(f"Device type '{device_type}' with version '{version}' not available") from e
    
    # Parse Apple Silicon device types
    elif device_type.startswith("apple."):
        backend_name = device_type[6:]  # Remove "apple." prefix
        
        # Handle version-specific backends
        if backend_name == "silicon":
            if version is not None:
                if isinstance(version, str):
                    # Convert string versions like "M1" to lowercase backend names
                    backend_name = version.lower().replace(" ", "")
                else:
                    # Handle numeric versions (unlikely for Apple Silicon)
                    backend_name = f"m{version}"
        
        try:
            return get_backend(backend_name, device_id=device_id)
        except (ValueError, ImportError) as e:
            raise ValueError(f"Device type '{device_type}' with version '{version}' not available") from e
    
    # Default fallback
    return Device(device_id=device_id, device_type=device_type)


def get_available_devices() -> List[Device]:
    """
    Get list of available devices.
    
    Returns:
        List of available devices
    """
    # Use capability detector to get available devices
    from .capability import detector
    available_devices = []
    
    for device_desc in detector.get_available_devices():
        device = detector.create_device(
            device_type=device_desc["type"],
            device_id=device_desc["device_id"],
            version=device_desc["version"]
        )
        available_devices.append(device)
    
    return available_devices


class DeviceContext:
    """
    Context manager for temporarily switching the active device.
    """
    
    def __init__(self, device_type: Optional[str] = None, device_id: Optional[int] = None,
                 version: Optional[Union[int, str]] = None):
        """
        Initialize device context.
        
        Args:
            device_type: Device type to use in the context
            device_id: Optional device ID
            version: Optional specific version of the device
        """
        self.device_type = device_type
        self.device_id = device_id
        self.version = version
        self.device = None
        self.previous_device = None
    
    def __enter__(self) -> Device:
        """
        Enter the context, activating the specified device.
        
        Returns:
            The activated device
        """
        # Import here to avoid circular imports
        import prism
        
        # Save the previous device
        self.previous_device = prism.get_active_device()
        
        # Create and set the new device
        self.device = get_device(self.device_type, self.device_id, self.version)
        prism.set_active_device(self.device)
        
        return self.device
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context, restoring the previous device.
        """
        # Import here to avoid circular imports
        import prism
        
        # Restore the previous device
        if self.previous_device is not None:
            prism.set_active_device(self.previous_device)