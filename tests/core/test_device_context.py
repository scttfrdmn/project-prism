"""
Tests for device context management.

These tests verify that the device context manager works correctly with all Graviton versions.
"""

import os
import unittest
from unittest import mock

import pytest

import src
from src.backends.graviton import (
    GravitonDevice,
    Graviton1Device,
    Graviton2Device,
    Graviton3Device,
    Graviton3EDevice,
    Graviton4Device
)
from src.core.device import Device, DeviceContext


class TestDeviceContext(unittest.TestCase):
    """Test the device context management functionality."""

    def setUp(self):
        """Set up test case."""
        # Save original active device
        self.original_device = src._active_device
        
        # Reset active device for tests
        src._active_device = None
    
    def tearDown(self):
        """Tear down test case."""
        # Restore original active device
        src._active_device = self.original_device
    
    def test_get_active_device_default(self):
        """Test get_active_device with default initialization."""
        with mock.patch('src.get_device', return_value=mock.MagicMock(spec=Device)) as mock_get_device:
            device = src.get_active_device()
            
            # Should initialize a default device
            mock_get_device.assert_called_once()
            self.assertIsNotNone(device)
    
    def test_set_active_device(self):
        """Test set_active_device."""
        mock_device = mock.MagicMock(spec=Device)
        src.set_active_device(mock_device)
        
        # Should set the active device
        self.assertEqual(src.get_active_device(), mock_device)
    
    def test_device_context_manager(self):
        """Test DeviceContext as a context manager."""
        # Create mock devices
        mock_device1 = mock.MagicMock(spec=Device)
        mock_device2 = mock.MagicMock(spec=Device)
        
        # Set initial active device
        src.set_active_device(mock_device1)
        
        # Mock get_device to return our second mock device
        with mock.patch('src.get_device', return_value=mock_device2):
            # Use context manager
            with src.device("aws.graviton", version=3):
                # Inside context, active device should be mock_device2
                self.assertEqual(src.get_active_device(), mock_device2)
            
            # Outside context, active device should be mock_device1 again
            self.assertEqual(src.get_active_device(), mock_device1)
    
    def test_graviton_context_managers(self):
        """Test specialized Graviton context managers."""
        # Create mock devices for each Graviton version
        mock_devices = {
            "1": mock.MagicMock(spec=Graviton1Device),
            "2": mock.MagicMock(spec=Graviton2Device),
            "3": mock.MagicMock(spec=Graviton3Device),
            "3E": mock.MagicMock(spec=Graviton3EDevice),
            "4": mock.MagicMock(spec=Graviton4Device),
        }
        
        # Set initial active device
        initial_device = mock.MagicMock(spec=Device)
        src.set_active_device(initial_device)
        
        # Test each Graviton version context manager
        with mock.patch('src.get_device', side_effect=lambda *args, **kwargs: 
                       mock_devices[str(kwargs.get('version', '1'))]):
            
            # Test Graviton 1
            with src.graviton1():
                self.assertEqual(src.get_active_device(), mock_devices["1"])
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test Graviton 2
            with src.graviton2():
                self.assertEqual(src.get_active_device(), mock_devices["2"])
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test Graviton 3
            with src.graviton3():
                self.assertEqual(src.get_active_device(), mock_devices["3"])
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test Graviton 3E
            with src.graviton3e():
                self.assertEqual(src.get_active_device(), mock_devices["3E"])
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test Graviton 4
            with src.graviton4():
                self.assertEqual(src.get_active_device(), mock_devices["4"])
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test generic Graviton with version parameter
            with src.graviton(version="3E"):
                self.assertEqual(src.get_active_device(), mock_devices["3E"])
            self.assertEqual(src.get_active_device(), initial_device)
    
    def test_trainium_inferentia_context_managers(self):
        """Test Trainium and Inferentia context managers."""
        # Create mock devices
        mock_trainium = mock.MagicMock(spec=Device)
        mock_inferentia = mock.MagicMock(spec=Device)
        
        # Set initial active device
        initial_device = mock.MagicMock(spec=Device)
        src.set_active_device(initial_device)
        
        # Test context managers
        with mock.patch('src.get_device', side_effect=lambda device_type, **kwargs: 
                      mock_trainium if device_type == "aws.trainium" else mock_inferentia):
            
            # Test Trainium
            with src.trainium():
                self.assertEqual(src.get_active_device(), mock_trainium)
            self.assertEqual(src.get_active_device(), initial_device)
            
            # Test Inferentia
            with src.inferentia():
                self.assertEqual(src.get_active_device(), mock_inferentia)
            self.assertEqual(src.get_active_device(), initial_device)
    
    def test_nested_context_managers(self):
        """Test nested context managers."""
        # Create mock devices
        mock_devices = {
            "graviton3": mock.MagicMock(spec=Graviton3Device),
            "graviton4": mock.MagicMock(spec=Graviton4Device),
            "trainium": mock.MagicMock(spec=Device)
        }
        
        # Set initial active device
        initial_device = mock.MagicMock(spec=Device)
        src.set_active_device(initial_device)
        
        # Define side effect for mock
        def get_device_side_effect(device_type=None, **kwargs):
            if device_type == "aws.graviton":
                if kwargs.get("version") == 3:
                    return mock_devices["graviton3"]
                elif kwargs.get("version") == 4:
                    return mock_devices["graviton4"]
            elif device_type == "aws.trainium":
                return mock_devices["trainium"]
            return initial_device
        
        # Test nested context managers
        with mock.patch('src.get_device', side_effect=get_device_side_effect):
            with src.graviton3():
                self.assertEqual(src.get_active_device(), mock_devices["graviton3"])
                
                # Nested context
                with src.graviton4():
                    self.assertEqual(src.get_active_device(), mock_devices["graviton4"])
                
                # Back to outer context
                self.assertEqual(src.get_active_device(), mock_devices["graviton3"])
                
                # Another nested context
                with src.trainium():
                    self.assertEqual(src.get_active_device(), mock_devices["trainium"])
                
                # Back to outer context again
                self.assertEqual(src.get_active_device(), mock_devices["graviton3"])
            
            # Back to initial device
            self.assertEqual(src.get_active_device(), initial_device)


if __name__ == "__main__":
    unittest.main()