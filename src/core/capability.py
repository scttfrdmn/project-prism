"""
Prism hardware capability detection and selection system.

This module provides functionality for automatically detecting available hardware
capabilities and selecting the appropriate backend based on workload characteristics.
"""

import logging
import os
import platform
import re
import subprocess
from typing import Dict, List, Optional, Union, Any, Tuple

from ..backends import get_backend

logger = logging.getLogger(__name__)


class CapabilityDetector:
    """
    Hardware capability detection system.
    
    This class provides methods for detecting available hardware accelerators
    and their capabilities, with specialized detection for AWS Silicon.
    """
    
    # Mapping of instance types to Graviton versions
    GRAVITON_INSTANCE_TYPES = {
        # Graviton 1
        'a1': 1,
        # Graviton 2
        'c6g': 2, 'm6g': 2, 'r6g': 2, 't4g': 2, 'x2gd': 2, 'c6gd': 2, 'm6gd': 2, 'r6gd': 2,
        # Graviton 3
        'c7g': 3, 'm7g': 3, 'r7g': 3, 'c7gd': 3, 'm7gd': 3, 'r7gd': 3,
        # Graviton 3E
        'c7gn': '3E', 
        # Graviton 4
        'c8g': 4, 'm8g': 4, 'r8g': 4
    }
    
    def __init__(self):
        """
        Initialize capability detector.
        """
        self.detected_devices = []
        self.aws_instance_type = self._detect_aws_instance_type()
        self.available_accelerators = {}
        
        # Detect capabilities
        self.detect_capabilities()
        
    def _detect_aws_instance_type(self) -> Optional[str]:
        """
        Detect AWS instance type.
        
        Returns:
            AWS instance type string or None if not on AWS
        """
        # Check environment variable (useful for testing)
        if "AWS_INSTANCE_TYPE" in os.environ:
            return os.environ["AWS_INSTANCE_TYPE"]
        
        # Try to read instance type from AWS metadata service
        try:
            # This URL is available on all AWS instances
            result = subprocess.run(
                ["curl", "-s", "http://169.254.169.254/latest/meta-data/instance-type"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
        
        # Not on AWS or couldn't determine instance type
        return None
    
    def _detect_neuron_devices(self) -> Dict[str, Any]:
        """
        Detect AWS Neuron (Trainium/Inferentia) devices.
        
        Returns:
            Dictionary with detected Neuron devices
        """
        neuron_devices = {
            "trainium": [],
            "inferentia": [],
            "version": {}
        }
        
        # Check if AWS_NEURON_SDK_VERSION is set (useful for testing)
        neuron_sdk_version = os.environ.get("AWS_NEURON_SDK_VERSION", "")
        
        # On a real system, we would use neuron-ls or other Neuron SDK tools
        # For now, infer from instance type
        if self.aws_instance_type:
            if self.aws_instance_type.startswith("trn1"):
                # Trainium instance
                count = 1
                if "32xl" in self.aws_instance_type:
                    count = 32
                elif "2xl" in self.aws_instance_type:
                    count = 2
                
                neuron_devices["trainium"] = list(range(count))
                neuron_devices["version"]["trainium"] = 1
                
            elif self.aws_instance_type.startswith("inf1"):
                # Inferentia 1 instance
                count = 1
                if "24xl" in self.aws_instance_type:
                    count = 16
                elif "6xl" in self.aws_instance_type:
                    count = 4
                elif "2xl" in self.aws_instance_type:
                    count = 1
                
                neuron_devices["inferentia"] = list(range(count))
                neuron_devices["version"]["inferentia"] = 1
                
            elif self.aws_instance_type.startswith("inf2"):
                # Inferentia 2 instance
                count = 1
                if "48xl" in self.aws_instance_type:
                    count = 12
                elif "24xl" in self.aws_instance_type:
                    count = 6
                elif "8xl" in self.aws_instance_type:
                    count = 2
                elif "xl" in self.aws_instance_type:
                    count = 1
                
                neuron_devices["inferentia"] = list(range(count))
                neuron_devices["version"]["inferentia"] = 2
        
        return neuron_devices
    
    def _detect_fpga_devices(self) -> Dict[str, Any]:
        """
        Detect AWS FPGA devices.
        
        Returns:
            Dictionary with detected FPGA devices
        """
        fpga_devices = {
            "devices": [],
            "instance_type": "",
        }
        
        # Infer from instance type
        if self.aws_instance_type and self.aws_instance_type.startswith("f1"):
            fpga_count = 1
            if "16xl" in self.aws_instance_type:
                fpga_count = 8
            elif "4xl" in self.aws_instance_type:
                fpga_count = 2
            
            fpga_devices["devices"] = list(range(fpga_count))
            fpga_devices["instance_type"] = self.aws_instance_type
        
        return fpga_devices
    
    def _detect_graviton_support(self) -> Dict[str, Any]:
        """
        Detect AWS Graviton support.
        
        Returns:
            Dictionary with detected Graviton capabilities
        """
        graviton_info = {
            "available": False,
            "version": 0,
            "architecture": platform.machine(),
        }
        
        # Force Graviton detection for testing
        if "AWS_GRAVITON_VERSION" in os.environ:
            version = os.environ["AWS_GRAVITON_VERSION"]
            if version in ('1', '2', '3', '3E', '4'):
                graviton_info["available"] = True
                graviton_info["version"] = version if version == '3E' else int(version)
                return graviton_info
        
        # Check if running on ARM architecture (Graviton)
        if platform.machine() == "aarch64":
            graviton_info["available"] = True
            
            # Infer Graviton version from instance type
            if self.aws_instance_type:
                # Extract the instance family prefix (e.g., 'c6g' from 'c6g.xlarge')
                instance_family = re.match(r'([a-z0-9]+)', self.aws_instance_type)
                if instance_family:
                    family = instance_family.group(1)
                    for prefix, version in self.GRAVITON_INSTANCE_TYPES.items():
                        if family.startswith(prefix):
                            graviton_info["version"] = version
                            return graviton_info
            
            # CPU hardware detection if instance type detection failed
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    
                # Graviton 1: Cortex-A72
                if "CPU part\t: 0xd08" in cpuinfo:
                    graviton_info["version"] = 1
                # Graviton 2: Neoverse N1
                elif "CPU part\t: 0xd0c" in cpuinfo:
                    graviton_info["version"] = 2
                # Graviton 3: Neoverse V1
                elif "CPU part\t: 0xd40" in cpuinfo:
                    # To differentiate between 3 and 3E would require more detailed information
                    # For now, assume 3
                    graviton_info["version"] = 3
                # Graviton 4: Future part ID
                elif "CPU part\t: 0xd4f" in cpuinfo:  # Placeholder part ID
                    graviton_info["version"] = 4
                # Default to Graviton 2 if we can't determine the exact version
                else:
                    graviton_info["version"] = 2
            except (IOError, FileNotFoundError):
                # If we can't read cpuinfo, default to Graviton 2
                graviton_info["version"] = 2
        
        return graviton_info
    
    def _detect_apple_silicon_support(self) -> Dict[str, Any]:
        """
        Detect Apple Silicon support.
        
        Returns:
            Dictionary with detected Apple Silicon capabilities
        """
        apple_silicon_info = {
            "available": False,
            "version": None,
            "architecture": platform.machine(),
        }
        
        # Check if running on macOS
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            return apple_silicon_info
        
        apple_silicon_info["available"] = True
        
        # Force version detection for testing
        if "APPLE_SILICON_VERSION" in os.environ:
            apple_silicon_info["version"] = os.environ["APPLE_SILICON_VERSION"]
            return apple_silicon_info
        
        # Get Mac model identifier
        try:
            model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode().strip()
            
            # Map model to version
            model_to_version = {
                # M1 family
                "MacBookAir10,1": "M1",
                "MacBookPro17,1": "M1",
                "MacBookPro18,3": "M1Pro",
                "MacBookPro18,4": "M1Max",
                "Mac13,2": "M1Ultra",
                # M2 family
                "Mac14,2": "M2",
                "MacBookPro19,1": "M2Pro",
                "MacBookPro19,2": "M2Max",
                "Mac14,13": "M2Ultra",
                # M3 family
                "Mac15,5": "M3",
                "MacBookPro21,1": "M3Pro",
                "MacBookPro21,2": "M3Max",
                "Mac15,12": "M3Ultra",
                # M4 family
                "iPad14,7": "M4",
                "Mac16,1": "M4Pro",
                "Mac16,2": "M4Max",
            }
            
            if model in model_to_version:
                apple_silicon_info["version"] = model_to_version[model]
                return apple_silicon_info
        except:
            pass
        
        # Alternative detection method
        try:
            sysctl_output = subprocess.check_output(["sysctl", "-a"]).decode()
            
            # Check for M4 (ARMv9 architecture)
            if "hw.optional.arm64_4" in sysctl_output:
                apple_silicon_info["version"] = "M4"
            # Check for M3 with specific GPU capabilities
            elif "hw.optional.amx_version" in sysctl_output and "3.0" in sysctl_output:
                apple_silicon_info["version"] = "M3"
            # Check for M2
            elif "hw.optional.armv8_2_sve" in sysctl_output:
                apple_silicon_info["version"] = "M2"
            # Default to M1 if nothing else matches
            else:
                apple_silicon_info["version"] = "M1"
        except:
            # Default to M1 as the conservative option
            apple_silicon_info["version"] = "M1"
        
        return apple_silicon_info
    
    def detect_capabilities(self) -> Dict[str, Any]:
        """
        Detect all available hardware capabilities.
        
        Returns:
            Dictionary with detected capabilities
        """
        logger.info("Detecting hardware capabilities")
        
        # Initialize capabilities
        self.available_accelerators = {
            "neuron": self._detect_neuron_devices(),
            "fpga": self._detect_fpga_devices(),
            "graviton": self._detect_graviton_support(),
            "apple_silicon": self._detect_apple_silicon_support(),
            "aws_instance_type": self.aws_instance_type
        }
        
        # Create device list based on detected accelerators
        self.detected_devices = []
        
        # Add Trainium devices
        for device_id in self.available_accelerators["neuron"]["trainium"]:
            version = self.available_accelerators["neuron"]["version"].get("trainium", 1)
            backend_name = f"trainium{version}"
            self.detected_devices.append({
                "type": "trainium",
                "device_id": device_id,
                "version": version,
                "backend": backend_name
            })
        
        # Add Inferentia devices
        for device_id in self.available_accelerators["neuron"]["inferentia"]:
            version = self.available_accelerators["neuron"]["version"].get("inferentia", 1)
            backend_name = f"inferentia{version}"
            self.detected_devices.append({
                "type": "inferentia",
                "device_id": device_id,
                "version": version,
                "backend": backend_name
            })
        
        # Add FPGA devices
        for device_id in self.available_accelerators["fpga"]["devices"]:
            self.detected_devices.append({
                "type": "fpga",
                "device_id": device_id,
                "version": 1,
                "backend": "fpga"
            })
        
        # Add Graviton if available
        if self.available_accelerators["graviton"]["available"]:
            # Get version and determine appropriate backend name
            version = self.available_accelerators["graviton"]["version"]
            version_str = str(version)
            
            # Map version to backend name
            if version_str == "1":
                backend_name = "graviton1"
            elif version_str == "2":
                backend_name = "graviton2"
            elif version_str == "3":
                backend_name = "graviton3"
            elif version_str == "3E":
                backend_name = "graviton3e"
            elif version_str == "4":
                backend_name = "graviton4"
            else:
                backend_name = "graviton"  # Default to generic backend
            
            self.detected_devices.append({
                "type": "graviton",
                "device_id": 0,
                "version": version,
                "backend": backend_name
            })
            
        # Add Apple Silicon if available
        if self.available_accelerators["apple_silicon"]["available"]:
            # Get version and determine appropriate backend name
            version = self.available_accelerators["apple_silicon"]["version"]
            
            # Map version to backend name
            if version.startswith("M1"):
                if version == "M1":
                    backend_name = "m1"
                elif version == "M1Pro":
                    backend_name = "m1pro"
                elif version == "M1Max":
                    backend_name = "m1max"
                elif version == "M1Ultra":
                    backend_name = "m1ultra"
                else:
                    backend_name = "m1"
            elif version.startswith("M2"):
                if version == "M2":
                    backend_name = "m2"
                elif version == "M2Pro":
                    backend_name = "m2pro"
                elif version == "M2Max":
                    backend_name = "m2max"
                elif version == "M2Ultra":
                    backend_name = "m2ultra"
                else:
                    backend_name = "m2"
            elif version.startswith("M3"):
                if version == "M3":
                    backend_name = "m3"
                elif version == "M3Pro":
                    backend_name = "m3pro"
                elif version == "M3Max":
                    backend_name = "m3max"
                elif version == "M3Ultra":
                    backend_name = "m3ultra"
                else:
                    backend_name = "m3"
            elif version.startswith("M4"):
                if version == "M4":
                    backend_name = "m4"
                elif version == "M4Pro":
                    backend_name = "m4pro"
                elif version == "M4Max":
                    backend_name = "m4max"
                else:
                    backend_name = "m4"
            else:
                backend_name = "apple_silicon"  # Default to generic backend
            
            self.detected_devices.append({
                "type": "apple_silicon",
                "device_id": 0,
                "version": version,
                "backend": backend_name
            })
        
        logger.info(f"Detected {len(self.detected_devices)} accelerator(s)")
        for device in self.detected_devices:
            logger.info(f"  - {device['type']} v{device['version']} (id={device['device_id']})")
        
        return self.available_accelerators
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """
        Get list of available devices.
        
        Returns:
            List of device descriptors
        """
        return self.detected_devices
    
    def get_device_count(self, device_type: str = None) -> int:
        """
        Get number of available devices.
        
        Args:
            device_type: Optional device type filter
            
        Returns:
            Number of available devices
        """
        if device_type is None:
            return len(self.detected_devices)
        
        return sum(1 for device in self.detected_devices if device["type"] == device_type)
    
    def create_device(self, 
                     device_type: str = None, 
                     device_id: int = 0,
                     version: Optional[Union[int, str]] = None) -> Any:
        """
        Create device object for a specific accelerator.
        
        Args:
            device_type: Device type (trainium, inferentia, fpga, graviton)
            device_id: Device ID
            version: Optional version specification
            
        Returns:
            Device object
            
        Raises:
            ValueError: If device is not available
        """
        # If no device_type specified, use first available device
        if device_type is None and self.detected_devices:
            device_desc = self.detected_devices[0]
            return get_backend(device_desc["backend"], device_id=device_desc["device_id"])
        
        # Find matching device
        for device_desc in self.detected_devices:
            if device_desc["type"] == device_type and device_desc["device_id"] == device_id:
                if version is None or str(device_desc["version"]) == str(version):
                    return get_backend(device_desc["backend"], device_id=device_id)
        
        # Handle the case where we're specifically asking for a Graviton version
        if device_type == "graviton" and version is not None:
            # Get backend name based on version
            version_str = str(version)
            if version_str == "1":
                backend_name = "graviton1"
            elif version_str == "2":
                backend_name = "graviton2"
            elif version_str == "3":
                backend_name = "graviton3"
            elif version_str == "3E":
                backend_name = "graviton3e"
            elif version_str == "4":
                backend_name = "graviton4"
            else:
                backend_name = "graviton"
                
            # Check if this backend is supported
            try:
                return get_backend(backend_name, device_id=device_id)
            except ValueError:
                pass  # Continue to error handling
        
        # Device not found
        err_msg = f"No {device_type} device with ID {device_id}"
        if version is not None:
            err_msg += f" and version {version}"
        err_msg += " available"
        
        raise ValueError(err_msg)
    

class BackendSelector:
    """
    Backend selection system based on workload characteristics.
    
    This class provides methods for selecting the most appropriate backend
    based on workload characteristics and available hardware.
    """
    
    def __init__(self, capability_detector: CapabilityDetector = None):
        """
        Initialize backend selector.
        
        Args:
            capability_detector: Optional capability detector instance
        """
        self.capability_detector = capability_detector or CapabilityDetector()
        
    def select_backend(self, workload_type: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Select appropriate backend for workload.
        
        Args:
            workload_type: Workload type (training, inference, hpc, etc.)
            **kwargs: Additional workload characteristics
            
        Returns:
            Tuple of (backend_name, device_config)
        """
        available_devices = self.capability_detector.get_available_devices()
        
        if not available_devices:
            logger.warning("No accelerators detected, falling back to CPU")
            return "cpu", {"device_id": 0}
        
        # Extract workload characteristics
        batch_size = kwargs.get("batch_size", 1)
        precision = kwargs.get("precision", "fp32")
        memory_usage = kwargs.get("memory_usage", "medium")  # low, medium, high
        model_size = kwargs.get("model_size", "medium")  # small, medium, large
        network_intensive = kwargs.get("network_intensive", False)
        ray_tracing = kwargs.get("ray_tracing", False)
        
        # Special case for ray tracing workloads
        if ray_tracing:
            # Check for M3 or M4 Apple Silicon
            apple_m3_plus_devices = [device for device in available_devices 
                               if device["type"] == "apple_silicon" and 
                               (device["version"].startswith("M3") or 
                                device["version"].startswith("M4"))]
            if apple_m3_plus_devices:
                device = apple_m3_plus_devices[0]
                return device["backend"], {"device_id": device["device_id"]}
        
        # Match workload to ideal backend
        if workload_type == "training":
            # For training, prefer Trainium
            trainium_devices = [device for device in available_devices if device["type"] == "trainium"]
            if trainium_devices:
                device = trainium_devices[0]
                return f"trainium{device['version']}", {"device_id": device["device_id"]}
            
            # If no Trainium, check for high-end Apple Silicon 
            apple_max_devices = [device for device in available_devices 
                           if device["type"] == "apple_silicon" and 
                           ("Max" in device["version"] or "Ultra" in device["version"])]
            if apple_max_devices:
                device = apple_max_devices[0]
                return device["backend"], {"device_id": device["device_id"]}
            
            # Fall back to other devices
            if any(device["type"] == "fpga" for device in available_devices):
                return "fpga", {"device_id": 0}
            
        elif workload_type == "inference":
            # For inference, prefer Inferentia
            inferentia_devices = [device for device in available_devices if device["type"] == "inferentia"]
            if inferentia_devices:
                device = inferentia_devices[0]
                # For large models or high batch size on Inferentia 1, prefer version 2 if available
                if model_size == "large" or batch_size > 64:
                    inferentia2_devices = [d for d in inferentia_devices if d["version"] == 2]
                    if inferentia2_devices:
                        device = inferentia2_devices[0]
                
                return f"inferentia{device['version']}", {"device_id": device["device_id"]}
            
            # Check for Apple Silicon with Neural Engine optimizations
            apple_m4_devices = [device for device in available_devices 
                          if device["type"] == "apple_silicon" and 
                          device["version"].startswith("M4")]
            if apple_m4_devices:
                device = apple_m4_devices[0]
                return device["backend"], {"device_id": device["device_id"]}
            
            apple_silicon_devices = [device for device in available_devices if device["type"] == "apple_silicon"]
            if apple_silicon_devices:
                # Sort by version (M4 > M3 > M2 > M1)
                apple_silicon_devices.sort(key=lambda d: d["version"], reverse=True)
                device = apple_silicon_devices[0]
                return device["backend"], {"device_id": device["device_id"]}
            
            # Fall back to other devices
            if any(device["type"] == "fpga" for device in available_devices):
                return "fpga", {"device_id": 0}
            
        elif workload_type == "hpc" or workload_type == "custom_logic":
            # For HPC, check for Apple Silicon M1/M2 Ultra or M3/M4 Max/Ultra first
            apple_hpc_devices = [device for device in available_devices 
                           if device["type"] == "apple_silicon" and 
                           (device["version"].endswith("Ultra") or 
                            (device["version"].startswith("M3") and device["version"].endswith("Max")) or
                            (device["version"].startswith("M4") and device["version"].endswith("Max")))]
            if apple_hpc_devices:
                device = apple_hpc_devices[0]
                return device["backend"], {"device_id": device["device_id"]}
                
            # For HPC or custom logic, prefer FPGA
            fpga_devices = [device for device in available_devices if device["type"] == "fpga"]
            if fpga_devices:
                device = fpga_devices[0]
                return "fpga", {"device_id": device["device_id"]}
                
            # For network-intensive workloads, prefer Graviton 3E if available
            if network_intensive:
                graviton_devices = [device for device in available_devices if device["type"] == "graviton"]
                graviton3e_devices = [d for d in graviton_devices if str(d["version"]) == "3E"]
                
                if graviton3e_devices:
                    return "graviton3e", {"device_id": 0}
        
        # Check for Apple Silicon
        apple_silicon_devices = [device for device in available_devices if device["type"] == "apple_silicon"]
        if apple_silicon_devices:
            # Sort by version (M4 > M3 > M2 > M1)
            apple_silicon_devices.sort(key=lambda d: d["version"], reverse=True)
            device = apple_silicon_devices[0]
            return device["backend"], {"device_id": device["device_id"]}
                
        # Fall back to Graviton if available, selecting the most appropriate version
        graviton_devices = [device for device in available_devices if device["type"] == "graviton"]
        if graviton_devices:
            # Select the highest version for general compute
            graviton_devices.sort(key=lambda d: str(d["version"]), reverse=True)
            device = graviton_devices[0]
            
            # Map version to backend name
            version = device["version"]
            backend_mapping = {
                1: "graviton1",
                2: "graviton2",
                3: "graviton3",
                "3E": "graviton3e",
                4: "graviton4"
            }
            
            backend_name = backend_mapping.get(version, "graviton")
            return backend_name, {"device_id": 0}
        
        # Default fallback to first available device
        device = available_devices[0]
        return device["backend"], {"device_id": device["device_id"]}
    
    def create_optimal_device(self, workload_type: str, **kwargs) -> Any:
        """
        Create optimal device for workload.
        
        Args:
            workload_type: Workload type (training, inference, hpc, etc.)
            **kwargs: Additional workload characteristics
            
        Returns:
            Optimal device instance
        """
        backend_name, device_config = self.select_backend(workload_type, **kwargs)
        
        # Add kwargs to device config
        device_config.update({k: v for k, v in kwargs.items() if k not in device_config})
        
        # Create and return device
        return get_backend(backend_name, **device_config)