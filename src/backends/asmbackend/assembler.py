"""
Assembly code assembler interface for the Prism framework.

This module provides classes for interfacing with system assemblers
such as GNU Assembler (as) and LLVM assembler to convert assembly
code to executable machine code.
"""

import os
import subprocess
import tempfile
import platform
from typing import Dict, List, Optional, Tuple, Union


class AssemblerError(Exception):
    """Exception raised for errors during assembly process."""
    pass


class Assembler:
    """Base class for assembly code assemblers."""
    
    def __init__(self, syntax: str = "att"):
        """
        Initialize the assembler.
        
        Args:
            syntax: Assembly syntax style ("att" or "intel" for x86_64, ignored for ARM)
        """
        self.syntax = syntax
        self._detect_assembler()
    
    def _detect_assembler(self) -> None:
        """Detect available assembler on the system."""
        self.assembler_path = None
        
        # Try common assembler paths
        assemblers = ["as", "llvm-as", "gas"]
        for assembler in assemblers:
            try:
                result = subprocess.run(
                    ["which", assembler], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    self.assembler_path = result.stdout.strip()
                    self.assembler_type = assembler
                    break
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
                
        if not self.assembler_path:
            raise AssemblerError("No suitable assembler found. Please install GNU Assembler or LLVM.")
    
    def assemble(self, asm_code: str, output_file: Optional[str] = None) -> str:
        """
        Assemble the given assembly code.
        
        Args:
            asm_code: Assembly code to assemble
            output_file: Optional path for the output object file
        
        Returns:
            Path to the assembled object file
        """
        raise NotImplementedError("Subclasses must implement this method")


class ARMAssembler(Assembler):
    """Assembler for ARM assembly code."""
    
    def assemble(self, asm_code: str, output_file: Optional[str] = None) -> str:
        """
        Assemble ARM assembly code.
        
        Args:
            asm_code: ARM assembly code to assemble
            output_file: Optional path for the output object file
        
        Returns:
            Path to the assembled object file
        """
        # Create temporary file for assembly code
        with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as asm_file:
            asm_file_path = asm_file.name
            asm_file.write(asm_code.encode('utf-8'))
        
        # Determine output path
        if not output_file:
            output_file = os.path.splitext(asm_file_path)[0] + ".o"
        
        # Determine ARM architecture flag
        arm_arch = "-march=armv8-a"
        if platform.machine() in ("arm64", "aarch64"):
            # Check if we have ARMv9 support
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                if "ARMv9" in cpuinfo:
                    arm_arch = "-march=armv9-a"
            except (FileNotFoundError, IOError):
                # Fall back to ARMv8-a on macOS or other platforms
                pass
        
        # Run assembler command
        cmd = [
            self.assembler_path,
            arm_arch,
            "-o", output_file,
            asm_file_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise AssemblerError(f"Assembly failed: {result.stderr}")
            
            # Clean up temporary files
            os.unlink(asm_file_path)
            
            return output_file
        except subprocess.SubprocessError as e:
            raise AssemblerError(f"Failed to run assembler: {str(e)}")


class X86Assembler(Assembler):
    """Assembler for x86_64 assembly code."""
    
    def assemble(self, asm_code: str, output_file: Optional[str] = None) -> str:
        """
        Assemble x86_64 assembly code.
        
        Args:
            asm_code: x86_64 assembly code to assemble
            output_file: Optional path for the output object file
        
        Returns:
            Path to the assembled object file
        """
        # Create temporary file for assembly code
        with tempfile.NamedTemporaryFile(suffix=".s", delete=False) as asm_file:
            asm_file_path = asm_file.name
            asm_file.write(asm_code.encode('utf-8'))
        
        # Determine output path
        if not output_file:
            output_file = os.path.splitext(asm_file_path)[0] + ".o"
        
        # Determine syntax flag for x86_64
        syntax_flag = "--" + self.syntax + "-syntax"
        
        # Run assembler command
        cmd = [
            self.assembler_path,
            syntax_flag,
            "-o", output_file,
            asm_file_path
        ]
        
        # For llvm-as, syntax is specified differently
        if self.assembler_type == "llvm-as":
            cmd = [
                self.assembler_path,
                "-o", output_file,
                asm_file_path
            ]
            # Prepend the syntax indicator in the assembly code
            if self.syntax == "intel":
                asm_code = ".intel_syntax noprefix\n" + asm_code
            else:
                asm_code = ".att_syntax\n" + asm_code
            with open(asm_file_path, "w") as f:
                f.write(asm_code)
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise AssemblerError(f"Assembly failed: {result.stderr}")
            
            # Clean up temporary files
            os.unlink(asm_file_path)
            
            return output_file
        except subprocess.SubprocessError as e:
            raise AssemblerError(f"Failed to run assembler: {str(e)}")


def get_assembler(arch: str, syntax: str = "att") -> Assembler:
    """
    Factory function to get the appropriate assembler for the target architecture.
    
    Args:
        arch: Target architecture ("arm", "aarch64", "x86_64")
        syntax: Assembly syntax style ("att" or "intel" for x86_64)
    
    Returns:
        Appropriate Assembler instance
    """
    if arch in ("arm", "aarch64", "arm64"):
        return ARMAssembler(syntax)
    elif arch in ("x86_64", "x86", "amd64"):
        return X86Assembler(syntax)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")