"""
Compiler interface for the C/C++ backend.

This module provides a wrapper around system compilers for compiling
C/C++ code with architecture-specific optimizations.
"""

import os
import sys
import tempfile
import platform
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Union

from ...core.capability import get_device_capabilities

logger = logging.getLogger(__name__)


class CompilerError(Exception):
    """Exception raised for errors during compilation."""
    pass


class CompilerWrapper:
    """
    Wrapper for C/C++ compilers with architecture-specific optimizations.
    
    This class provides an interface for compiling C/C++ code using
    system compilers (GCC, Clang, MSVC) with appropriate optimization
    flags for different processor architectures.
    """
    
    def __init__(self, 
                 compiler: str = None,
                 arch: str = None,
                 optimization_level: str = "-O3",
                 cpp: bool = True):
        """
        Initialize the compiler wrapper.
        
        Args:
            compiler: Compiler to use (auto-detected if None)
            arch: Target architecture (auto-detected if None)
            optimization_level: Optimization level to use
            cpp: Whether to compile C++ (True) or C (False) code
        """
        self.arch = arch or self._detect_architecture()
        self.cpp = cpp
        self.optimization_level = optimization_level
        self.compiler = compiler or self._detect_compiler()
        self.capabilities = get_device_capabilities()
        
        # Generate flags based on architecture and capabilities
        self.arch_flags = self._generate_arch_flags()
    
    def _detect_architecture(self) -> str:
        """
        Detect the current system architecture.
        
        Returns:
            Architecture identifier
        """
        machine = platform.machine().lower()
        if machine in ("arm", "aarch64", "arm64"):
            return "arm64"
        elif machine in ("x86_64", "amd64", "x64"):
            return "x86_64"
        else:
            raise ValueError(f"Unsupported architecture: {machine}")
    
    def _detect_compiler(self) -> str:
        """
        Detect an available C/C++ compiler on the system.
        
        Returns:
            Path to the compiler executable
        """
        # Compiler preference order (preferred compilers first)
        if sys.platform == 'darwin':
            # macOS
            if self.cpp:
                compilers = ['clang++', 'g++', 'icpx', 'armclang++']
            else:
                compilers = ['clang', 'gcc', 'icx', 'armclang']
        elif sys.platform == 'win32':
            # Windows
            if self.cpp:
                compilers = ['cl', 'g++', 'clang++', 'icpx', 'armclang++', 'clang++.exe', 'aocl-clang++']
            else:
                compilers = ['cl', 'gcc', 'clang', 'icx', 'armclang', 'clang.exe', 'aocl-clang']
        else:
            # Linux and others
            if self.cpp:
                compilers = ['g++', 'clang++', 'icpx', 'armclang++', 'aocl-clang++']
            else:
                compilers = ['gcc', 'clang', 'icx', 'armclang', 'aocl-clang']
        
        # Special paths to search for specialized compilers
        special_paths = {
            'icx': [
                '/opt/intel/oneapi/compiler/latest/bin/icx',
                'C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin\\icx.exe',
            ],
            'icpx': [
                '/opt/intel/oneapi/compiler/latest/bin/icpx',
                'C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\latest\\bin\\icpx.exe',
            ],
            'armclang': [
                '/opt/arm/arm-compiler/bin/armclang',
                'C:\\Program Files\\ARM\\Compiler\\bin\\armclang.exe',
            ],
            'armclang++': [
                '/opt/arm/arm-compiler/bin/armclang++',
                'C:\\Program Files\\ARM\\Compiler\\bin\\armclang++.exe',
            ],
            'aocl-clang': [
                '/opt/AMD/aocc/bin/aocl-clang',
                'C:\\Program Files\\AMD\\AOCC\\bin\\aocl-clang.exe',
            ],
            'aocl-clang++': [
                '/opt/AMD/aocc/bin/aocl-clang++',
                'C:\\Program Files\\AMD\\AOCC\\bin\\aocl-clang++.exe',
            ],
        }
        
        # Try to find a compiler
        for compiler in compilers:
            # First check if it's in PATH
            try:
                if sys.platform == 'win32' and compiler == 'cl':
                    # Special case for MSVC
                    result = subprocess.run(
                        ['where', compiler], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True,
                        check=False
                    )
                else:
                    result = subprocess.run(
                        ['which', compiler], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True,
                        check=False
                    )
                
                if result.returncode == 0:
                    path = result.stdout.strip().split('\n')[0]
                    logger.info(f"Using compiler: {path}")
                    return path
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            # If not in PATH, check special paths for specialized compilers
            if compiler in special_paths:
                for path in special_paths[compiler]:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        logger.info(f"Using compiler: {path}")
                        return path
        
        # Check for Intel oneAPI environment script and source it if available
        if sys.platform != 'win32':  # Unix/Linux/macOS
            intel_vars = "/opt/intel/oneapi/setvars.sh"
            if os.path.exists(intel_vars):
                try:
                    # Source Intel oneAPI environment and check for compilers again
                    logger.info("Found Intel oneAPI environment, activating...")
                    result = subprocess.run(
                        [f"source {intel_vars} && which icpx" if self.cpp else f"source {intel_vars} && which icx"],
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )
                    if result.returncode == 0:
                        path = result.stdout.strip()
                        logger.info(f"Using Intel compiler: {path}")
                        return path
                except subprocess.SubprocessError:
                    pass
        
        raise CompilerError("No suitable compiler found. Please install GCC, Clang, Intel oneAPI, ARM Compiler, or AMD AOCC.")
    
    def _generate_arch_flags(self) -> List[str]:
        """
        Generate architecture-specific compiler flags.
        
        Returns:
            List of compiler flags
        """
        flags = []
        compiler_type = self._detect_compiler_type()
        
        if self.arch == "arm64":
            # ARM flags
            if compiler_type == "msvc":
                # MSVC
                flags.append("/arch:arm64")
            elif compiler_type == "armcc":
                # ARM Compiler
                flags.append("-march=armv8-a")
                
                # Add NEON flags
                flags.append("-mfpu=neon")
                
                # Add SVE flags if supported
                if self.capabilities.get("sve_supported", False):
                    flags.append("-march=armv8.2-a+sve")
                    
            else:
                # GCC/Clang/Intel/AMD
                flags.append("-march=native")
                
                # Add NEON flags
                flags.append("-mfpu=neon")
                
                # Add SVE flags if supported
                if self.capabilities.get("sve_supported", False):
                    flags.append("-march=armv8.2-a+sve")
                    
        elif self.arch == "x86_64":
            # x86_64 flags
            if compiler_type == "msvc":
                # MSVC
                if self.capabilities.get("avx512_supported", False):
                    flags.append("/arch:AVX512")
                elif self.capabilities.get("avx2_supported", False):
                    flags.append("/arch:AVX2")
                elif self.capabilities.get("avx_supported", False):
                    flags.append("/arch:AVX")
            elif compiler_type == "icc":
                # Intel Compiler
                flags.append("-xHost")  # Optimize for current CPU architecture
                
                # Add specific instruction set flags
                if self.capabilities.get("avx512_supported", False):
                    flags.append("-mavx512f")
                elif self.capabilities.get("avx2_supported", False):
                    flags.append("-mavx2")
                elif self.capabilities.get("avx_supported", False):
                    flags.append("-mavx")
                    
                # Intel-specific optimizations
                flags.append("-qopt-zmm-usage=high")  # Aggressively use AVX-512 registers
                
            elif compiler_type == "aocc":
                # AMD AOCC
                flags.append("-march=native")
                
                # AMD-specific optimizations
                flags.append("-mafma")  # AMD FMA instructions
                
                # Add specific instruction set flags
                if self.capabilities.get("avx2_supported", False):
                    flags.append("-mavx2")
                elif self.capabilities.get("avx_supported", False):
                    flags.append("-mavx")
                    
            else:
                # GCC/Clang
                flags.append("-march=native")
                
                # Add specific instruction set flags
                if self.capabilities.get("avx512_supported", False):
                    flags.append("-mavx512f")
                elif self.capabilities.get("avx2_supported", False):
                    flags.append("-mavx2")
                elif self.capabilities.get("avx_supported", False):
                    flags.append("-mavx")
        
        return flags
        
    def _detect_compiler_type(self) -> str:
        """
        Detect the type of compiler being used.
        
        Returns:
            Compiler type: "gcc", "clang", "msvc", "icc", "armcc", or "aocc"
        """
        compiler_path = self.compiler.lower()
        
        if "cl" in compiler_path or "msvc" in compiler_path:
            return "msvc"
        elif "icx" in compiler_path or "icpx" in compiler_path or "icc" in compiler_path or "intel" in compiler_path:
            return "icc"
        elif "armclang" in compiler_path:
            return "armcc"
        elif "aocl" in compiler_path or "aocc" in compiler_path:
            return "aocc" 
        elif "clang" in compiler_path:
            return "clang"
        else:
            # Default to GCC
            return "gcc"
    
    def _get_compiler_flags(self, shared: bool = True) -> List[str]:
        """
        Get compiler flags for the current architecture and configuration.
        
        Args:
            shared: Whether to compile as a shared library
        
        Returns:
            List of compiler flags
        """
        flags = []
        compiler_type = self._detect_compiler_type()
        
        # Optimization level
        if compiler_type == "msvc":
            # MSVC
            if self.optimization_level == "-O0":
                flags.append("/Od")
            elif self.optimization_level == "-O1":
                flags.append("/O1")
            elif self.optimization_level == "-O2":
                flags.append("/O2")
            elif self.optimization_level == "-O3":
                flags.append("/Ox")
        elif compiler_type == "icc":
            # Intel Compiler
            flags.append(self.optimization_level)
            if self.optimization_level == "-O3":
                # Intel-specific optimizations for maximum performance
                flags.append("-ipo")       # Interprocedural optimization
                flags.append("-qopt-mem-layout-trans=3")  # Aggressive memory layout optimizations
                flags.append("-qopt-report=5")  # Generate optimization reports
        elif compiler_type == "armcc":
            # ARM Compiler
            if self.optimization_level == "-O0":
                flags.append("-O0")
            elif self.optimization_level == "-O1":
                flags.append("-O1")
            elif self.optimization_level == "-O2":
                flags.append("-O2")
            elif self.optimization_level == "-O3":
                flags.append("-O3")
                flags.append("-Otime")     # Optimize for speed over size
                flags.append("-ffast-math") # Fast math operations
        elif compiler_type == "aocc":
            # AMD AOCC (based on Clang)
            flags.append(self.optimization_level)
            if self.optimization_level == "-O3":
                # AMD-specific optimizations
                flags.append("-ffast-math")
                flags.append("-funroll-loops")
                flags.append("-ftree-vectorize")
                flags.append("-mllvm -polly")  # Polly - LLVM's polyhedral optimizer
        else:
            # GCC/Clang
            flags.append(self.optimization_level)
        
        # Architecture-specific flags
        flags.extend(self.arch_flags)
        
        # Shared library flags
        if shared:
            if compiler_type == "msvc":
                # MSVC
                flags.append("/LD")
            elif compiler_type == "armcc":
                # ARM Compiler
                flags.append("-shared")
                flags.append("-fPIC")
            elif compiler_type == "icc":
                # Intel Compiler
                flags.append("-shared")
                flags.append("-fPIC")
                if sys.platform == 'win32':
                    flags.append("/dll")
            else:
                # GCC/Clang/AOCC
                flags.append("-shared")
                flags.append("-fPIC")
        
        # Language standard
        if self.cpp:
            if compiler_type == "msvc":
                # MSVC
                flags.append("/std:c++17")
            elif compiler_type == "armcc":
                # ARM Compiler
                flags.append("-std=c++17")
            elif compiler_type == "icc":
                # Intel Compiler
                if sys.platform == 'win32':
                    flags.append("/std:c++17")
                else:
                    flags.append("-std=c++17")
            else:
                # GCC/Clang/AOCC
                flags.append("-std=c++17")
        else:
            if compiler_type == "msvc":
                # MSVC
                flags.append("/std:c11")
            elif compiler_type == "armcc":
                # ARM Compiler
                flags.append("-std=c11")
            elif compiler_type == "icc":
                # Intel Compiler
                if sys.platform == 'win32':
                    flags.append("/std:c11")
                else:
                    flags.append("-std=c11")
            else:
                # GCC/Clang/AOCC
                flags.append("-std=c11")
        
        # Other flags based on compiler type
        if compiler_type == "gcc":
            # GCC-specific flags
            flags.append("-ffast-math")
            flags.append("-mfma")  # Use FMA instructions if available
            flags.append("-ftree-vectorize")
            flags.append("-fomit-frame-pointer")
        elif compiler_type == "clang":
            # Clang-specific flags
            flags.append("-ffast-math")
            flags.append("-mfma")  # Use FMA instructions if available
            flags.append("-fomit-frame-pointer")
        elif compiler_type == "icc":
            # Intel-specific flags
            if sys.platform != 'win32':
                flags.append("-ffast-math")
                flags.append("-fomit-frame-pointer")
        elif compiler_type == "aocc":
            # AMD-specific flags
            flags.append("-mfma")  # AMD FMA instructions
            flags.append("-flto")  # Link-time optimization
            flags.append("-fomit-frame-pointer")
        elif compiler_type == "armcc":
            # ARM-specific flags
            flags.append("-ffast-math")
            flags.append("-ffp-contract=fast")  # Contract floating-point operations
            flags.append("-fomit-frame-pointer")
            
        return flags
    
    def compile(self, 
               source_code: str, 
               output_file: Optional[str] = None, 
               shared: bool = True) -> str:
        """
        Compile C/C++ source code.
        
        Args:
            source_code: C/C++ source code to compile
            output_file: Path to the output file (auto-generated if None)
            shared: Whether to compile as a shared library
        
        Returns:
            Path to the compiled library
        """
        # Create a temporary file for the source code
        with tempfile.NamedTemporaryFile(
            suffix=".cpp" if self.cpp else ".c", 
            delete=False
        ) as src_file:
            src_path = src_file.name
            src_file.write(source_code.encode('utf-8'))
        
        try:
            # Determine output path
            if output_file is None:
                if sys.platform == 'darwin':
                    ext = '.dylib' if shared else '.o'
                elif sys.platform == 'win32':
                    ext = '.dll' if shared else '.obj'
                else:
                    ext = '.so' if shared else '.o'
                output_file = os.path.splitext(src_path)[0] + ext
            
            # Get compiler flags
            flags = self._get_compiler_flags(shared)
            
            # Build the command
            if "cl" in self.compiler:
                # MSVC
                cmd = [
                    self.compiler,
                    src_path,
                    "/Fe" + output_file,
                    *flags
                ]
            else:
                # GCC/Clang
                cmd = [
                    self.compiler,
                    src_path,
                    "-o", output_file,
                    *flags
                ]
            
            # Run the compiler
            logger.debug(f"Compiler command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                error_msg = f"Compilation failed with command {' '.join(cmd)}: {result.stderr}"
                logger.error(error_msg)
                raise CompilerError(error_msg)
            
            logger.info(f"Compilation successful: {output_file}")
            return output_file
            
        finally:
            # Clean up the source file
            try:
                os.unlink(src_path)
            except (OSError, IOError):
                pass
    
    def get_compiler_version(self) -> str:
        """
        Get the version of the compiler.
        
        Returns:
            Compiler version string
        """
        try:
            if "cl" in self.compiler:
                # MSVC
                result = subprocess.run(
                    [self.compiler],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                # Extract version from the output
                for line in result.stderr.split('\n'):
                    if "Microsoft" in line and "Compiler" in line:
                        return line.strip()
                return "Microsoft Visual C++ (version unknown)"
            else:
                # GCC/Clang
                result = subprocess.run(
                    [self.compiler, "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                # Return first line of the version output
                return result.stdout.strip().split('\n')[0]
        except (subprocess.SubprocessError, FileNotFoundError, IndexError):
            return "Unknown compiler version"