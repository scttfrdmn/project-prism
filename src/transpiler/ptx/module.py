"""
PTX module representation.

This module defines the internal representation of PTX code,
including modules, functions, instructions, and registers.
"""

from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
import re

from .types import PTXType, PTXRegisterType, PTXMemorySpace, PTXAddressSpace


@dataclass
class PTXRegister:
    """PTX register representation."""
    name: str
    reg_type: PTXType
    is_vector: bool = False
    vector_size: int = 1
    is_array: bool = False
    array_size: Optional[int] = None
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation of register
        """
        result = f"%{self.name}"
        if self.is_vector:
            result += f"<{self.vector_size}>"
        if self.is_array:
            result += f"[{self.array_size}]" if self.array_size else "[]"
        return result


@dataclass
class PTXOperand:
    """PTX instruction operand."""
    text: str  # Original text representation
    is_register: bool = False
    is_immediate: bool = False
    is_label: bool = False
    is_predicate: bool = False
    register: Optional[PTXRegister] = None
    immediate_value: Optional[Union[int, float]] = None
    label: Optional[str] = None
    
    @classmethod
    def from_text(cls, text: str) -> 'PTXOperand':
        """
        Create operand from text representation.
        
        Args:
            text: Text representation of operand
            
        Returns:
            PTXOperand instance
        """
        text = text.strip()
        
        # Check if it's a register
        if text.startswith('%'):
            return cls(text=text, is_register=True)
        # Check if it's a predicate
        elif text.startswith('@'):
            return cls(text=text, is_predicate=True)
        # Check if it's a label
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:$', text):
            return cls(text=text, is_label=True, label=text[:-1])
        # Check if it's an immediate integer
        elif re.match(r'^-?\d+$', text):
            return cls(text=text, is_immediate=True, immediate_value=int(text))
        # Check if it's an immediate float
        elif re.match(r'^-?\d+\.\d*f?$', text):
            return cls(text=text, is_immediate=True, immediate_value=float(text.rstrip('f')))
        # Check if it's a hexadecimal immediate
        elif re.match(r'^0[xX][0-9a-fA-F]+$', text):
            return cls(text=text, is_immediate=True, immediate_value=int(text, 16))
        # Otherwise, it could be a label reference or other identifier
        else:
            return cls(text=text, is_label=True, label=text)


@dataclass
class PTXInstruction:
    """PTX instruction representation."""
    opcode: str
    type_suffix: Optional[PTXType] = None
    operands: List[PTXOperand] = field(default_factory=list)
    predicate: Optional[PTXOperand] = None
    options: Dict[str, str] = field(default_factory=dict)
    line_number: Optional[int] = None
    original_text: Optional[str] = None
    
    @property
    def has_side_effects(self) -> bool:
        """
        Check if instruction has side effects (memory writes, etc.).
        
        Returns:
            True if instruction has side effects, False otherwise
        """
        # Memory store operations
        if self.opcode.startswith('st.'):
            return True
        # Atomic operations
        if self.opcode.startswith('atom.'):
            return True
        # Memory fence operations
        if self.opcode in ('membar', 'fence'):
            return True
        # Control flow that could affect state
        if self.opcode in ('ret', 'exit'):
            return True
        return False
    
    @property
    def is_branch(self) -> bool:
        """
        Check if instruction is a branch.
        
        Returns:
            True if instruction is a branch, False otherwise
        """
        return self.opcode in ('bra', 'call', 'ret', 'exit', 'brx') or self.opcode.startswith('bra.')
    
    @property
    def is_memory_access(self) -> bool:
        """
        Check if instruction accesses memory.
        
        Returns:
            True if instruction accesses memory, False otherwise
        """
        return self.opcode.startswith('ld.') or self.opcode.startswith('st.') or self.opcode.startswith('atom.')
    
    @property
    def is_compute(self) -> bool:
        """
        Check if instruction performs computation.
        
        Returns:
            True if instruction performs computation, False otherwise
        """
        compute_opcodes = {
            'add', 'sub', 'mul', 'div', 'rem', 'mad', 'fma',
            'min', 'max', 'abs', 'neg', 'not', 'and', 'or', 'xor',
            'shl', 'shr', 'cvt', 'mov', 'selp', 'setp'
        }
        # Check if opcode without type suffix is in compute_opcodes
        opcode_base = self.opcode.split('.')[0]
        return opcode_base in compute_opcodes
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation of instruction
        """
        parts = [self.opcode]
        if self.type_suffix:
            parts.append(self.type_suffix.value)
        for option, value in self.options.items():
            parts.append(f"{option}")
            if value:
                parts.append(value)
        
        result = f"{'.'.join(parts)}"
        if self.predicate:
            result = f"@{self.predicate.text} {result}"
        
        operand_str = ", ".join(op.text for op in self.operands)
        if operand_str:
            result = f"{result} {operand_str}"
            
        return result


@dataclass
class PTXFunction:
    """PTX function representation."""
    name: str
    parameters: List[PTXRegister] = field(default_factory=list)
    return_type: Optional[PTXType] = None
    instructions: List[PTXInstruction] = field(default_factory=list)
    local_registers: Dict[str, PTXRegister] = field(default_factory=dict)
    labels: Dict[str, int] = field(default_factory=dict)  # label -> instruction index
    called_functions: Set[str] = field(default_factory=set)
    is_kernel: bool = False
    
    def add_instruction(self, instruction: PTXInstruction) -> None:
        """
        Add instruction to function.
        
        Args:
            instruction: PTX instruction to add
        """
        self.instructions.append(instruction)
        
        # If this is a call instruction, record the called function
        if instruction.opcode == 'call':
            if len(instruction.operands) >= 1 and instruction.operands[0].is_label:
                self.called_functions.add(instruction.operands[0].label)
    
    def set_label(self, label: str) -> None:
        """
        Set label at current instruction position.
        
        Args:
            label: Label name
        """
        self.labels[label] = len(self.instructions)
    
    def add_register(self, register: PTXRegister) -> None:
        """
        Add local register to function.
        
        Args:
            register: Register to add
        """
        self.local_registers[register.name] = register
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation of function
        """
        kind = "kernel" if self.is_kernel else "func"
        signature = f".visible .{kind} {self.name}"
        
        param_strs = []
        for param in self.parameters:
            param_type = f".{param.reg_type.value}"
            if param.is_vector:
                param_type += f".v{param.vector_size}"
            param_strs.append(f".param {param_type} {param}")
        
        param_str = "(\n  " + ",\n  ".join(param_strs) + "\n)" if param_strs else "()"
        
        if self.return_type:
            result_str = f" .{self.return_type.value}"
        else:
            result_str = ""
        
        body_lines = []
        for idx, instr in enumerate(self.instructions):
            for label, pos in self.labels.items():
                if pos == idx:
                    body_lines.append(f"{label}:")
            body_lines.append(f"  {instr}")
            
        body_str = "\n".join(body_lines)
        
        return f"{signature}{param_str}{result_str} {{\n{body_str}\n}}"


@dataclass
class PTXModule:
    """PTX module representation."""
    version: Tuple[int, int] = (7, 8)  # Default to PTX ISA version 7.8
    target: str = "sm_86"  # Default to Compute Capability 8.6
    target_address_size: int = 64  # Default to 64-bit
    functions: Dict[str, PTXFunction] = field(default_factory=dict)
    global_variables: Dict[str, Any] = field(default_factory=dict)
    textures: Dict[str, Any] = field(default_factory=dict)
    samplers: Dict[str, Any] = field(default_factory=dict)
    surface: Dict[str, Any] = field(default_factory=dict)
    
    def add_function(self, function: PTXFunction) -> None:
        """
        Add function to module.
        
        Args:
            function: PTX function to add
        """
        self.functions[function.name] = function
    
    def __str__(self) -> str:
        """
        Get string representation.
        
        Returns:
            String representation of module
        """
        version_str = f".version {self.version[0]}.{self.version[1]}"
        target_str = f".target {self.target}"
        address_size_str = f".address_size {self.target_address_size}"
        
        header = f"{version_str}\n{target_str}\n{address_size_str}\n"
        
        # Add global variables
        globals_str = ""
        for name, var in self.global_variables.items():
            globals_str += f".global {var}\n"
        
        # Add functions
        functions_str = "\n\n".join(str(func) for func in self.functions.values())
        
        return f"{header}\n{globals_str}\n{functions_str}"