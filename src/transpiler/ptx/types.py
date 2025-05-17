"""
PTX type system definitions.

This module defines the type system used in NVIDIA PTX,
which will be translated to AWS Neuron IR types.
"""

import enum
from typing import Dict, List, Optional, Union


class PTXType(enum.Enum):
    """Basic types in PTX."""
    PRED = "pred"  # Predicate register (1-bit boolean)
    B8 = "b8"      # 8-bit byte
    B16 = "b16"    # 16-bit byte
    B32 = "b32"    # 32-bit byte
    B64 = "b64"    # 64-bit byte
    U8 = "u8"      # 8-bit unsigned integer
    U16 = "u16"    # 16-bit unsigned integer
    U32 = "u32"    # 32-bit unsigned integer
    U64 = "u64"    # 64-bit unsigned integer
    S8 = "s8"      # 8-bit signed integer
    S16 = "s16"    # 16-bit signed integer
    S32 = "s32"    # 32-bit signed integer
    S64 = "s64"    # 64-bit signed integer
    F16 = "f16"    # 16-bit floating point
    F32 = "f32"    # 32-bit floating point
    F64 = "f64"    # 64-bit floating point
    FP8E4M3 = "fp8e4m3"   # 8-bit floating point (e4m3 format)
    FP8E5M2 = "fp8e5m2"   # 8-bit floating point (e5m2 format)
    BF16 = "bf16"  # 16-bit brain floating point
    
    @classmethod
    def from_string(cls, s: str) -> 'PTXType':
        """
        Convert string representation to PTXType.
        
        Args:
            s: String representation of PTX type
            
        Returns:
            Corresponding PTXType enum value
            
        Raises:
            ValueError: If string doesn't match any PTX type
        """
        for t in cls:
            if t.value == s:
                return t
        raise ValueError(f"Unknown PTX type: {s}")
    
    @property
    def bit_width(self) -> int:
        """
        Get bit width of this type.
        
        Returns:
            Bit width as integer
        """
        if self in (PTXType.B8, PTXType.U8, PTXType.S8, 
                  PTXType.FP8E4M3, PTXType.FP8E5M2):
            return 8
        elif self in (PTXType.B16, PTXType.U16, PTXType.S16, 
                    PTXType.F16, PTXType.BF16):
            return 16
        elif self in (PTXType.B32, PTXType.U32, PTXType.S32, PTXType.F32):
            return 32
        elif self in (PTXType.B64, PTXType.U64, PTXType.S64, PTXType.F64):
            return 64
        elif self == PTXType.PRED:
            return 1
        else:
            raise ValueError(f"Cannot determine bit width for {self}")
    
    @property
    def is_float(self) -> bool:
        """
        Check if this is a floating point type.
        
        Returns:
            True if floating point, False otherwise
        """
        return self in (
            PTXType.F16, PTXType.F32, PTXType.F64, 
            PTXType.BF16, PTXType.FP8E4M3, PTXType.FP8E5M2
        )
    
    @property
    def is_integer(self) -> bool:
        """
        Check if this is an integer type.
        
        Returns:
            True if integer, False otherwise
        """
        return self in (
            PTXType.U8, PTXType.U16, PTXType.U32, PTXType.U64,
            PTXType.S8, PTXType.S16, PTXType.S32, PTXType.S64
        )
    
    @property
    def is_signed(self) -> bool:
        """
        Check if this is a signed type.
        
        Returns:
            True if signed, False otherwise
        """
        return self in (
            PTXType.S8, PTXType.S16, PTXType.S32, PTXType.S64,
            PTXType.F16, PTXType.F32, PTXType.F64, 
            PTXType.BF16, PTXType.FP8E4M3, PTXType.FP8E5M2
        )


class PTXVectorType:
    """
    Vector type in PTX (e.g., .v2, .v4).
    """
    
    def __init__(self, base_type: PTXType, dimensions: int):
        """
        Initialize vector type.
        
        Args:
            base_type: Base element type
            dimensions: Number of vector elements (e.g., 2, 4)
        """
        self.base_type = base_type
        self.dimensions = dimensions
    
    @property
    def bit_width(self) -> int:
        """
        Get total bit width of this vector type.
        
        Returns:
            Total bit width as integer
        """
        return self.base_type.bit_width * self.dimensions
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation (e.g., "f32.v4")
        """
        return f"{self.base_type.value}.v{self.dimensions}"


class PTXRegisterType(enum.Enum):
    """Register types in PTX."""
    REG = "reg"          # General purpose register
    SREG = "sreg"        # Special register
    CONST = "const"      # Constant memory
    GLOBAL = "global"    # Global memory
    LOCAL = "local"      # Local memory
    PARAM = "param"      # Parameter
    SHARED = "shared"    # Shared memory
    TEX = "tex"          # Texture memory
    SAMPLER = "sampler"  # Sampler


class PTXMemorySpace(enum.Enum):
    """Memory spaces in PTX."""
    GLOBAL = "global"      # Global memory
    LOCAL = "local"        # Local memory
    SHARED = "shared"      # Shared memory
    CONSTANT = "const"     # Constant memory
    PARAM = "param"        # Parameter memory
    GENERIC = "generic"    # Generic memory
    
    @classmethod
    def from_string(cls, s: str) -> 'PTXMemorySpace':
        """
        Convert string representation to PTXMemorySpace.
        
        Args:
            s: String representation of PTX memory space
            
        Returns:
            Corresponding PTXMemorySpace enum value
            
        Raises:
            ValueError: If string doesn't match any PTX memory space
        """
        for space in cls:
            if space.value == s:
                return space
        raise ValueError(f"Unknown PTX memory space: {s}")


class PTXAddressSpace(enum.Enum):
    """Address spaces in PTX."""
    GLOBAL = "global"      # Global address space
    LOCAL = "local"        # Local address space
    SHARED = "shared"      # Shared address space
    CONST = "const"        # Constant address space
    PARAM = "param"        # Parameter address space
    GENERIC = "generic"    # Generic address space
    
    @classmethod
    def from_string(cls, s: str) -> 'PTXAddressSpace':
        """
        Convert string representation to PTXAddressSpace.
        
        Args:
            s: String representation of PTX address space
            
        Returns:
            Corresponding PTXAddressSpace enum value
            
        Raises:
            ValueError: If string doesn't match any PTX address space
        """
        for space in cls:
            if space.value == s:
                return space
        raise ValueError(f"Unknown PTX address space: {s}")