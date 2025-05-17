"""
Prism core module for hardware abstraction and runtime management.
"""

from .device import Device
from .capability import CapabilityDetector, BackendSelector

__version__ = "0.1.0"

# Initialize capability detector
detector = CapabilityDetector()

# Initialize backend selector
selector = BackendSelector(detector)

# Convenience function to create optimal device
def create_device(workload_type="inference", **kwargs):
    """
    Create optimal device for workload.
    
    Args:
        workload_type: Workload type (training, inference, hpc, etc.)
        **kwargs: Additional workload characteristics
        
    Returns:
        Optimal device instance
    """
    return selector.create_optimal_device(workload_type, **kwargs)