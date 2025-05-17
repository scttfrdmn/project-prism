"""
Tests for the device module.
"""

import pytest
from prism.core.device import Device, get_device, get_available_devices


def test_device_initialization():
    """Test device initialization."""
    device = Device(device_id=0, device_type="test")
    assert device.device_id == 0
    assert device.device_type == "test"


def test_get_device():
    """Test get_device function."""
    device = get_device(device_type="test", device_id=0)
    assert device.device_type == "test"
    assert device.device_id == 0


def test_get_available_devices():
    """Test get_available_devices function."""
    devices = get_available_devices()
    assert isinstance(devices, list)