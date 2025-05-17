"""
ARM assembly code generation.

This module generates optimized assembly code for ARM architectures,
including ARMv8-A with NEON, SVE, and SVE2 extensions.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union, TextIO

from ...core.lowlevel.operation import OperationGraph, Operation, OperationType, DataType

logger = logging.getLogger(__name__)


class ARMAssemblyGenerator:
    """Generates assembly code for ARM architectures."""
    
    def __init__(self, 
                 dialect: str = "gnu", 
                 instruction_sets: Optional[List[str]] = None,
                 sve_vector_length: int = 256):
        """
        Initialize ARM assembly generator.
        
        Args:
            dialect: Assembly dialect (gnu, arm)
            instruction_sets: Instruction sets to use (neon, sve, sve2)
            sve_vector_length: SVE vector length in bits (128, 256, 512)
        """
        self.dialect = dialect.lower()
        self.instruction_sets = instruction_sets or ["neon"]
        self.sve_vector_length = sve_vector_length
        
        # Set of available instructions
        self.available_instructions = self._get_available_instructions()
    
    def _get_available_instructions(self) -> Dict[str, List[str]]:
        """
        Get available instructions for target architecture.
        
        Returns:
            Dictionary of instruction sets and their instructions
        """
        instructions = {}
        
        # Base ARMv8-A instructions
        instructions["base"] = [
            "add", "sub", "mul", "div", "ldr", "str", "mov", "and", "orr", "eor",
            "cmp", "b", "bl", "ret", "cbz", "cbnz"
        ]
        
        # NEON instructions
        if "neon" in self.instruction_sets:
            instructions["neon"] = [
                "ld1", "st1", "add", "sub", "mul", "fmla", "fmls", "fadd", "fsub", "fmul",
                "fcmp", "mov", "dup", "ext", "zip", "unzip", "trn"
            ]
        
        # SVE instructions
        if "sve" in self.instruction_sets:
            instructions["sve"] = [
                "ld1w", "st1w", "ptrue", "pfalse", "whilelt", "add", "sub", "mul", "fadd", "fsub",
                "fmul", "fmla", "fmls", "fcmp", "mov", "dup", "zip", "unzip"
            ]
            
        # SVE2 instructions
        if "sve2" in self.instruction_sets:
            instructions["sve2"] = [
                "ld1w", "st1w", "ptrue", "pfalse", "whilelt", "add", "sub", "mul", "fadd", "fsub",
                "fmul", "fmla", "fmls", "fcmp", "mov", "dup", "zip", "unzip", "bext", "bcax",
                "eor3", "xar"
            ]
            
        return instructions
    
    def _generate_function_prologue(self, func_name: str, out: TextIO) -> None:
        """
        Generate function prologue.
        
        Args:
            func_name: Function name
            out: Output file-like object
        """
        if self.dialect == "gnu":
            # GNU assembly format
            out.write(f".text\n")
            out.write(f".global {func_name}\n")
            out.write(f".type {func_name}, %function\n")
            out.write(f"{func_name}:\n")
            
            # Function prologue
            out.write("\t// Function prologue\n")
            out.write("\tstp x29, x30, [sp, #-16]!\n")
            out.write("\tmov x29, sp\n")
            
        elif self.dialect == "arm":
            # ARM assembly format
            out.write(f"\tAREA |.text|, CODE, READONLY\n")
            out.write(f"\tEXPORT {func_name}\n")
            out.write(f"{func_name}\n")
            
            # Function prologue
            out.write("\t; Function prologue\n")
            out.write("\tPUSH {r4-r11, lr}\n")
            
        else:
            # Default to GNU format
            out.write(f".text\n")
            out.write(f".global {func_name}\n")
            out.write(f"{func_name}:\n")
            
            # Function prologue
            out.write("\t// Function prologue\n")
            out.write("\tstp x29, x30, [sp, #-16]!\n")
            out.write("\tmov x29, sp\n")
    
    def _generate_function_epilogue(self, out: TextIO) -> None:
        """
        Generate function epilogue.
        
        Args:
            out: Output file-like object
        """
        if self.dialect == "gnu":
            # GNU assembly format
            out.write("\t// Function epilogue\n")
            out.write("\tldp x29, x30, [sp], #16\n")
            out.write("\tret\n")
            
        elif self.dialect == "arm":
            # ARM assembly format
            out.write("\t; Function epilogue\n")
            out.write("\tPOP {r4-r11, pc}\n")
            
        else:
            # Default to GNU format
            out.write("\t// Function epilogue\n")
            out.write("\tldp x29, x30, [sp], #16\n")
            out.write("\tret\n")
    
    def _generate_operation_neon(self, op: Operation, out: TextIO) -> None:
        """
        Generate NEON assembly code for an operation.
        
        Args:
            op: Operation to generate code for
            out: Output file-like object
        """
        if op.op_type == OperationType.VECTOR_ADD:
            # Vector addition using NEON
            inputs = [inp.name if isinstance(inp, Operation) else inp for inp in op.inputs]
            
            if len(inputs) != 2:
                raise ValueError(f"VECTOR_ADD operation expects 2 inputs, got {len(inputs)}")
            
            # Get vector dimension from attributes
            vector_size = op.attributes.get('size', 4)
            
            if op.output_type == DataType.FLOAT32:
                # 32-bit floating point vector addition
                out.write(f"\t// NEON Vector Addition: {op.name} = {inputs[0]} + {inputs[1]}\n")
                
                # Load vectors
                out.write(f"\tldr x0, ={inputs[0]}\n")
                out.write(f"\tldr x1, ={inputs[1]}\n")
                out.write(f"\tldr x2, ={op.name}\n")
                
                # Process 4 elements at a time
                out.write(f"\tmov x3, #{vector_size // 4}\n")
                out.write(f"1:\n")
                out.write(f"\tld1 {{v0.4s}}, [x0], #16\n")
                out.write(f"\tld1 {{v1.4s}}, [x1], #16\n")
                out.write(f"\tfadd v2.4s, v0.4s, v1.4s\n")
                out.write(f"\tst1 {{v2.4s}}, [x2], #16\n")
                out.write(f"\tsubs x3, x3, #1\n")
                out.write(f"\tbne 1b\n")
                
            elif op.output_type == DataType.FLOAT64:
                # 64-bit floating point vector addition
                out.write(f"\t// NEON Vector Addition: {op.name} = {inputs[0]} + {inputs[1]}\n")
                
                # Load vectors
                out.write(f"\tldr x0, ={inputs[0]}\n")
                out.write(f"\tldr x1, ={inputs[1]}\n")
                out.write(f"\tldr x2, ={op.name}\n")
                
                # Process 2 elements at a time
                out.write(f"\tmov x3, #{vector_size // 2}\n")
                out.write(f"1:\n")
                out.write(f"\tld1 {{v0.2d}}, [x0], #16\n")
                out.write(f"\tld1 {{v1.2d}}, [x1], #16\n")
                out.write(f"\tfadd v2.2d, v0.2d, v1.2d\n")
                out.write(f"\tst1 {{v2.2d}}, [x2], #16\n")
                out.write(f"\tsubs x3, x3, #1\n")
                out.write(f"\tbne 1b\n")
                
            else:
                # Integer vector addition
                out.write(f"\t// NEON Integer Vector Addition not implemented yet\n")
                
        elif op.op_type == OperationType.MATRIX_MULTIPLY:
            # Matrix multiplication using NEON
            inputs = [inp.name if isinstance(inp, Operation) else inp for inp in op.inputs]
            
            if len(inputs) != 2:
                raise ValueError(f"MATRIX_MULTIPLY operation expects 2 inputs, got {len(inputs)}")
            
            # Get matrix dimensions from attributes
            m = op.attributes.get('m', 4)
            n = op.attributes.get('n', 4)
            k = op.attributes.get('k', 4)
            
            out.write(f"\t// NEON Matrix Multiplication not implemented yet\n")
            
        else:
            # Default implementation for other operations
            out.write(f"\t// Operation {op.op_type.value} not implemented for NEON\n")
    
    def _generate_operation_sve(self, op: Operation, out: TextIO) -> None:
        """
        Generate SVE assembly code for an operation.
        
        Args:
            op: Operation to generate code for
            out: Output file-like object
        """
        if op.op_type == OperationType.VECTOR_ADD:
            # Vector addition using SVE
            inputs = [inp.name if isinstance(inp, Operation) else inp for inp in op.inputs]
            
            if len(inputs) != 2:
                raise ValueError(f"VECTOR_ADD operation expects 2 inputs, got {len(inputs)}")
            
            # Get vector dimension from attributes
            vector_size = op.attributes.get('size', 4)
            
            if op.output_type == DataType.FLOAT32:
                # 32-bit floating point vector addition
                out.write(f"\t// SVE Vector Addition: {op.name} = {inputs[0]} + {inputs[1]}\n")
                
                # Load vectors
                out.write(f"\tldr x0, ={inputs[0]}\n")
                out.write(f"\tldr x1, ={inputs[1]}\n")
                out.write(f"\tldr x2, ={op.name}\n")
                
                # Create predicate for vector length
                out.write(f"\tmov x3, #{vector_size}\n")
                out.write(f"\tptrue p0.s, vl{self.sve_vector_length // 32}\n")
                out.write(f"\twhilelt p1.s, xzr, x3\n")
                
                # Load vectors with predication
                out.write(f"\tld1w {{z0.s}}, p1/z, [x0]\n")
                out.write(f"\tld1w {{z1.s}}, p1/z, [x1]\n")
                
                # Add vectors
                out.write(f"\tfadd z2.s, p1/m, z0.s, z1.s\n")
                
                # Store result
                out.write(f"\tst1w {{z2.s}}, p1, [x2]\n")
                
            elif op.output_type == DataType.FLOAT64:
                # 64-bit floating point vector addition
                out.write(f"\t// SVE Vector Addition: {op.name} = {inputs[0]} + {inputs[1]}\n")
                
                # Load vectors
                out.write(f"\tldr x0, ={inputs[0]}\n")
                out.write(f"\tldr x1, ={inputs[1]}\n")
                out.write(f"\tldr x2, ={op.name}\n")
                
                # Create predicate for vector length
                out.write(f"\tmov x3, #{vector_size}\n")
                out.write(f"\tptrue p0.d, vl{self.sve_vector_length // 64}\n")
                out.write(f"\twhilelt p1.d, xzr, x3\n")
                
                # Load vectors with predication
                out.write(f"\tld1d {{z0.d}}, p1/z, [x0]\n")
                out.write(f"\tld1d {{z1.d}}, p1/z, [x1]\n")
                
                # Add vectors
                out.write(f"\tfadd z2.d, p1/m, z0.d, z1.d\n")
                
                # Store result
                out.write(f"\tst1d {{z2.d}}, p1, [x2]\n")
                
            else:
                # Integer vector addition
                out.write(f"\t// SVE Integer Vector Addition not implemented yet\n")
                
        elif op.op_type == OperationType.MATRIX_MULTIPLY:
            # Matrix multiplication using SVE
            inputs = [inp.name if isinstance(inp, Operation) else inp for inp in op.inputs]
            
            if len(inputs) != 2:
                raise ValueError(f"MATRIX_MULTIPLY operation expects 2 inputs, got {len(inputs)}")
            
            # Get matrix dimensions from attributes
            m = op.attributes.get('m', 4)
            n = op.attributes.get('n', 4)
            k = op.attributes.get('k', 4)
            
            out.write(f"\t// SVE Matrix Multiplication not implemented yet\n")
            
        else:
            # Default implementation for other operations
            out.write(f"\t// Operation {op.op_type.value} not implemented for SVE\n")
    
    def generate_assembly(self, graph: OperationGraph) -> str:
        """
        Generate assembly code from operation graph.
        
        Args:
            graph: Computational operation graph
            
        Returns:
            Generated assembly code as string
        """
        import io
        out = io.StringIO()
        
        # Add comments and metadata
        out.write(f"// ARM Assembly for {graph.name}\n")
        out.write(f"// Generated by Prism ARM Assembly Backend\n")
        out.write(f"// Dialect: {self.dialect}\n")
        out.write(f"// Instruction sets: {', '.join(self.instruction_sets)}\n\n")
        
        # Add data section
        out.write(f".data\n")
        
        # Declare input and output variables
        for input_name, input_type in graph.inputs:
            # Size would be determined from the graph in a real implementation
            size = 32  # Placeholder
            out.write(f"{input_name}:\n")
            out.write(f"\t.space {size}\n")
            
        for output_name in graph.outputs:
            # Size would be determined from the graph in a real implementation
            size = 32  # Placeholder
            out.write(f"{output_name}:\n")
            out.write(f"\t.space {size}\n")
            
        # Add text section with function
        self._generate_function_prologue(graph.name, out)
        
        # Generate operation code
        for op in graph.operations:
            # Skip operations that don't generate code
            if op.op_type == OperationType.LOAD or op.op_type == OperationType.STORE:
                continue
                
            # Choose instruction set to use
            if "sve2" in self.instruction_sets:
                # Use SVE2 instructions (not implemented yet)
                self._generate_operation_sve(op, out)
            elif "sve" in self.instruction_sets:
                # Use SVE instructions
                self._generate_operation_sve(op, out)
            elif "neon" in self.instruction_sets:
                # Use NEON instructions
                self._generate_operation_neon(op, out)
            else:
                # Use basic ARM instructions (not implemented)
                out.write(f"\t// Basic ARM implementation not provided for {op.op_type.value}\n")
        
        # Add function epilogue
        self._generate_function_epilogue(out)
        
        return out.getvalue()