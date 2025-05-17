"""
Tests for AWS Graviton backends.

These tests validate the functionality of all Graviton versions.
"""

import os
import unittest
from unittest import mock

import pytest

from src.backends.graviton import (
    GravitonDevice,
    Graviton1Device,
    Graviton2Device,
    Graviton3Device,
    Graviton3EDevice,
    Graviton4Device
)


class TestGravitonBaseDevice(unittest.TestCase):
    """Test the base GravitonDevice class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with mock.patch.object(GravitonDevice, '_detect_graviton_version', return_value=2):
            device = GravitonDevice()
            self.assertEqual(device.version, 2)
            self.assertEqual(device.version_str, "2")
    
    def test_init_custom_version(self):
        """Test initialization with custom version."""
        with mock.patch.object(GravitonDevice, '_detect_graviton_version', return_value=2):
            device = GravitonDevice(version=3)
            self.assertEqual(device.version, 3)
            self.assertEqual(device.version_str, "3")
    
    def test_detect_graviton_from_env(self):
        """Test detection of Graviton version from environment variable."""
        with mock.patch.dict(os.environ, {"AWS_GRAVITON_VERSION": "3"}):
            device = GravitonDevice()
            self.assertEqual(device.version, 3)
    
    def test_detect_graviton_3e_from_env(self):
        """Test detection of Graviton 3E from environment variable."""
        with mock.patch.dict(os.environ, {"AWS_GRAVITON_VERSION": "3E"}):
            device = GravitonDevice()
            self.assertEqual(device.version, "3E")
            self.assertEqual(device.version_str, "3E")
    
    def test_get_capabilities(self):
        """Test getting device capabilities."""
        with mock.patch.object(GravitonDevice, '_detect_graviton_version', return_value=2):
            device = GravitonDevice()
            capabilities = device.get_capabilities()
            self.assertEqual(capabilities["name"], "AWS Graviton 2")
            self.assertEqual(capabilities["architecture"], "arm64")
            self.assertEqual(capabilities["version"], "2")


class TestGravitonVersions(unittest.TestCase):
    """Test specific Graviton version implementations."""

    def test_graviton1_device(self):
        """Test Graviton 1 specific implementation."""
        device = Graviton1Device()
        self.assertEqual(device.version, 1)
        self.assertEqual(device.features["vector_width_bits"], 128)
        self.assertEqual(device.features["simd_ext"], ["NEON"])
        self.assertEqual(device.features["cores_per_socket"], 16)
    
    def test_graviton2_device(self):
        """Test Graviton 2 specific implementation."""
        device = Graviton2Device()
        self.assertEqual(device.version, 2)
        self.assertEqual(device.features["vector_width_bits"], 128)
        self.assertEqual(device.features["simd_ext"], ["NEON", "dotprod"])
        self.assertEqual(device.features["cores_per_socket"], 64)
    
    def test_graviton3_device(self):
        """Test Graviton 3 specific implementation."""
        device = Graviton3Device()
        self.assertEqual(device.version, 3)
        self.assertEqual(device.features["vector_width_bits"], 256)
        self.assertTrue("SVE" in device.features["simd_ext"])
        self.assertTrue(device.features["supports_sve"])
        self.assertEqual(device.features["sve_vector_length"], 256)
        self.assertEqual(device.features["cores_per_socket"], 64)
    
    def test_graviton3e_device(self):
        """Test Graviton 3E specific implementation."""
        device = Graviton3EDevice()
        self.assertEqual(device.version, "3E")
        self.assertEqual(device.features["vector_width_bits"], 256)  # Same as Graviton 3
        self.assertTrue("SVE" in device.features["simd_ext"])
        self.assertTrue(device.features["supports_sve"])
        self.assertEqual(device.features["sve_vector_length"], 256)  # Same as Graviton 3
        self.assertEqual(device.features["vector_performance_boost"], 35)  # 35% faster than G3
        self.assertTrue(device.features["hpc_optimized"] or device.features["enhanced_networking"])
    
    def test_graviton4_device(self):
        """Test Graviton 4 specific implementation."""
        device = Graviton4Device()
        self.assertEqual(device.version, 4)
        self.assertEqual(device.features["vector_width_bits"], 512)  # 512-bit vectors
        self.assertTrue("SVE2" in device.features["simd_ext"])
        self.assertTrue(device.features["supports_sve"])
        self.assertTrue(device.features["supports_sve2"])
        self.assertEqual(device.features["sve_vector_length"], 512)
        self.assertEqual(device.features["cores_per_socket"], 96)


class TestGravitonSpecializedFunctions(unittest.TestCase):
    """Test specialized functionality for different Graviton versions."""

    def test_sve_configuration(self):
        """Test SVE configuration retrieval."""
        # Graviton 2 doesn't support SVE
        g2 = Graviton2Device()
        self.assertEqual(g2.get_sve_configuration(), {})
        
        # Graviton 3 supports SVE
        g3 = Graviton3Device()
        sve_config = g3.get_sve_configuration()
        self.assertEqual(sve_config["vector_length"], 256)
        self.assertTrue(sve_config["supports_bf16"])
        self.assertFalse("supports_sve2" in sve_config)
        
        # Graviton 4 supports both SVE and SVE2
        g4 = Graviton4Device()
        sve_config = g4.get_sve_configuration()
        self.assertEqual(sve_config["vector_length"], 512)
        self.assertTrue(sve_config["supports_bf16"])
        self.assertTrue(sve_config["supports_sve2"])
    
    def test_graviton3e_specialized_functions(self):
        """Test Graviton 3E specialized functions."""
        g3e = Graviton3EDevice()
        
        # Test HPC optimization
        self.assertTrue(g3e.is_hpc_optimized())
        g3e.optimize_for_hpc(hpc_type="mpi")
        self.assertTrue("collective_operations" in g3e.features["current_optimizations"])
        
        # Test networking optimization
        self.assertTrue(g3e.is_network_optimized())
        g3e.optimize_for_networking(network_type="throughput")
        self.assertTrue("packet_batching" in g3e.features["current_optimizations"])
    
    def test_graviton4_confidential_computing(self):
        """Test Graviton 4 confidential computing feature."""
        g4 = Graviton4Device()
        self.assertTrue(g4.features["confidential_computing"])
        self.assertTrue(g4.enable_confidential_computing())
        self.assertTrue(g4.features["confidential_computing_enabled"])


class TestGravitonMemoryHandling(unittest.TestCase):
    """Test memory handling functions across Graviton versions."""

    def test_optimal_memory_alignment(self):
        """Test optimal memory alignment calculation."""
        # Graviton 1 & 2: 128-bit vectors = 16 bytes
        g1 = Graviton1Device()
        self.assertEqual(g1.get_optimal_memory_alignment(), 64)  # Max of vector width and cache line
        
        # Graviton 3 & 3E: 256-bit vectors = 32 bytes
        g3 = Graviton3Device()
        self.assertEqual(g3.get_optimal_memory_alignment(), 64)  # Max of vector width and cache line
        
        # Graviton 4: 512-bit vectors = 64 bytes
        g4 = Graviton4Device()
        self.assertEqual(g4.get_optimal_memory_alignment(), 64)  # Equal to both vector width and cache line
    
    def test_memory_allocation(self):
        """Test memory allocation."""
        g3 = Graviton3Device()
        buffer = g3.allocate_memory(1024)
        self.assertEqual(len(buffer), 1024)
        
        # Test copying data
        src_data = bytearray([1, 2, 3, 4])
        g3.copy_to_device(src_data, buffer, 4)
        self.assertEqual(buffer[:4], src_data)


if __name__ == "__main__":
    unittest.main()