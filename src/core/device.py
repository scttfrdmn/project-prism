"""
Device abstractions for hardware management.
"""

from typing import List, Optional, Dict, Any, Tuple


class Device:
    """
    Abstract representation of a compute device.
    """
    
    def __init__(self, device_id: Optional[int] = None, device_type: Optional[str] = None):
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
    

def get_device(device_type: Optional[str] = None, device_id: Optional[int] = None) -> Device:
    """
    Get a device instance of the specified type.
    
    Args:
        device_type: Device type to get
        device_id: Optional device ID
        
    Returns:
        Device instance
    """
    # This would be replaced with actual device selection logic
    return Device(device_id=device_id, device_type=device_type)


def get_available_devices() -> List[Device]:
    """
    Get list of available devices.
    
    Returns:
        List of available devices
    """
    # This would be replaced with actual device discovery logic
    return []