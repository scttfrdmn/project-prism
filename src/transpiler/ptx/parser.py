"""
PTX parser for analyzing NVIDIA PTX code.

This module provides a parser for NVIDIA PTX code that constructs
an internal representation for further processing and translation.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union, Any, TextIO
import logging

from .module import PTXModule, PTXFunction, PTXInstruction, PTXOperand, PTXRegister
from .types import PTXType, PTXRegisterType, PTXMemorySpace, PTXAddressSpace


logger = logging.getLogger(__name__)


class PTXParseError(Exception):
    """Exception raised for PTX parsing errors."""
    
    def __init__(self, message: str, line_number: int, line: str):
        """
        Initialize error.
        
        Args:
            message: Error message
            line_number: Line number where error occurred
            line: Line content
        """
        super().__init__(f"{message} at line {line_number}: {line}")
        self.message = message
        self.line_number = line_number
        self.line = line


class PTXParser:
    """
    Parser for NVIDIA PTX code.
    
    This class parses PTX code and constructs an internal representation
    for further processing and translation.
    """
    
    def __init__(self):
        """Initialize parser."""
        self.current_function: Optional[PTXFunction] = None
        self.module = PTXModule()
        self.line_number = 0
    
    def parse(self, code: str) -> PTXModule:
        """
        Parse PTX code.
        
        Args:
            code: PTX code to parse
            
        Returns:
            Parsed PTXModule
            
        Raises:
            PTXParseError: If parsing fails
        """
        self.module = PTXModule()
        self.current_function = None
        self.line_number = 0
        
        lines = code.splitlines()
        for i, line in enumerate(lines):
            self.line_number = i + 1
            self._parse_line(line)
            
        return self.module
    
    def parse_file(self, file_path: str) -> PTXModule:
        """
        Parse PTX code from file.
        
        Args:
            file_path: Path to PTX file
            
        Returns:
            Parsed PTXModule
            
        Raises:
            PTXParseError: If parsing fails
            FileNotFoundError: If file not found
        """
        with open(file_path, 'r') as f:
            code = f.read()
        return self.parse(code)
    
    def _parse_line(self, line: str) -> None:
        """
        Parse single line of PTX code.
        
        Args:
            line: Line to parse
            
        Raises:
            PTXParseError: If parsing fails
        """
        # Remove comments
        if '//' in line:
            line = line[:line.index('//')]
        
        # Skip empty lines
        line = line.strip()
        if not line:
            return
        
        try:
            # Check for different line types
            if line.startswith('.version'):
                self._parse_version(line)
            elif line.startswith('.target'):
                self._parse_target(line)
            elif line.startswith('.address_size'):
                self._parse_address_size(line)
            elif line.startswith('.visible') and ('.func' in line or '.kernel' in line):
                self._parse_function_start(line)
            elif line == '}' and self.current_function:
                self.module.add_function(self.current_function)
                self.current_function = None
            elif line.endswith(':') and self.current_function:
                # It's a label
                label = line[:-1].strip()
                self.current_function.set_label(label)
            elif self.current_function:
                # Inside function - parse instruction or directive
                if line.startswith('.'):
                    self._parse_directive(line)
                else:
                    self._parse_instruction(line)
            elif line.startswith('.global') or line.startswith('.shared') or line.startswith('.const'):
                self._parse_global_variable(line)
            else:
                logger.warning(f"Ignoring unknown line at {self.line_number}: {line}")
        except Exception as e:
            if not isinstance(e, PTXParseError):
                raise PTXParseError(str(e), self.line_number, line) from e
            raise
    
    def _parse_version(self, line: str) -> None:
        """
        Parse PTX version directive.
        
        Args:
            line: Line containing version directive
            
        Raises:
            PTXParseError: If parsing fails
        """
        match = re.match(r'\.version\s+(\d+)\.(\d+)', line)
        if not match:
            raise PTXParseError("Invalid .version directive", self.line_number, line)
        
        major = int(match.group(1))
        minor = int(match.group(2))
        self.module.version = (major, minor)
    
    def _parse_target(self, line: str) -> None:
        """
        Parse PTX target directive.
        
        Args:
            line: Line containing target directive
            
        Raises:
            PTXParseError: If parsing fails
        """
        match = re.match(r'\.target\s+([a-zA-Z0-9_]+)', line)
        if not match:
            raise PTXParseError("Invalid .target directive", self.line_number, line)
        
        self.module.target = match.group(1)
    
    def _parse_address_size(self, line: str) -> None:
        """
        Parse PTX address_size directive.
        
        Args:
            line: Line containing address_size directive
            
        Raises:
            PTXParseError: If parsing fails
        """
        match = re.match(r'\.address_size\s+(\d+)', line)
        if not match:
            raise PTXParseError("Invalid .address_size directive", self.line_number, line)
        
        self.module.target_address_size = int(match.group(1))
    
    def _parse_function_start(self, line: str) -> None:
        """
        Parse function start.
        
        Args:
            line: Line containing function declaration
            
        Raises:
            PTXParseError: If parsing fails
        """
        # Check if it's a kernel or function
        is_kernel = '.kernel' in line
        func_type = 'kernel' if is_kernel else 'func'
        
        # Extract function name
        match = re.search(fr'\.{func_type}\s+([a-zA-Z0-9_]+)', line)
        if not match:
            raise PTXParseError(f"Invalid .{func_type} directive", self.line_number, line)
        
        name = match.group(1)
        
        # Create new function
        self.current_function = PTXFunction(name=name, is_kernel=is_kernel)
        
        # Parse return type if present
        return_match = re.search(r'[\)\s]+\.(b|u|s|f)(\d+)\s*{', line)
        if return_match:
            type_prefix = return_match.group(1)
            type_bits = return_match.group(2)
            try:
                return_type = PTXType.from_string(f"{type_prefix}{type_bits}")
                self.current_function.return_type = return_type
            except ValueError:
                logger.warning(f"Unknown return type: {type_prefix}{type_bits}")
        
        # TODO: Parse parameters - this would require more sophisticated parsing
        # as parameters can span multiple lines
    
    def _parse_directive(self, line: str) -> None:
        """
        Parse directive inside function.
        
        Args:
            line: Line containing directive
            
        Raises:
            PTXParseError: If parsing fails
        """
        if not self.current_function:
            raise PTXParseError("Directive outside function", self.line_number, line)
        
        # Parse .reg directive for register declaration
        if line.startswith('.reg'):
            self._parse_reg_directive(line)
        # Other directives can be added here
    
    def _parse_reg_directive(self, line: str) -> None:
        """
        Parse .reg directive for register declaration.
        
        Args:
            line: Line containing .reg directive
            
        Raises:
            PTXParseError: If parsing fails
        """
        if not self.current_function:
            raise PTXParseError(".reg directive outside function", self.line_number, line)
        
        # Simple example: .reg .b32 %r1;
        match = re.match(r'\.reg\s+\.([a-zA-Z0-9]+)\s+(%[a-zA-Z0-9_]+)(?:\[(\d+)\])?;?', line)
        if not match:
            raise PTXParseError("Invalid .reg directive", self.line_number, line)
        
        type_str = match.group(1)
        reg_name = match.group(2)[1:]  # Remove % prefix
        array_size = int(match.group(3)) if match.group(3) else None
        
        try:
            reg_type = PTXType.from_string(type_str)
        except ValueError:
            # Try to handle vector types like .v4.f32
            vector_match = re.match(r'v(\d+)\.([a-zA-Z0-9]+)', type_str)
            if vector_match:
                vector_size = int(vector_match.group(1))
                base_type_str = vector_match.group(2)
                try:
                    reg_type = PTXType.from_string(base_type_str)
                    reg = PTXRegister(
                        name=reg_name,
                        reg_type=reg_type,
                        is_vector=True,
                        vector_size=vector_size,
                        is_array=array_size is not None,
                        array_size=array_size
                    )
                    self.current_function.add_register(reg)
                    return
                except ValueError:
                    raise PTXParseError(f"Unknown vector element type: {base_type_str}", 
                                       self.line_number, line)
            raise PTXParseError(f"Unknown register type: {type_str}", self.line_number, line)
        
        reg = PTXRegister(
            name=reg_name,
            reg_type=reg_type,
            is_array=array_size is not None,
            array_size=array_size
        )
        self.current_function.add_register(reg)
    
    def _parse_global_variable(self, line: str) -> None:
        """
        Parse global variable declaration.
        
        Args:
            line: Line containing global variable declaration
            
        Raises:
            PTXParseError: If parsing fails
        """
        # TODO: Implement global variable parsing
        # This is a placeholder that just logs the directive
        logger.debug(f"Global variable declaration: {line}")
    
    def _parse_instruction(self, line: str) -> None:
        """
        Parse instruction.
        
        Args:
            line: Line containing instruction
            
        Raises:
            PTXParseError: If parsing fails
        """
        if not self.current_function:
            raise PTXParseError("Instruction outside function", self.line_number, line)
        
        # Handle predicated instruction
        predicate = None
        if '@' in line and not line.strip().startswith('@'):
            # Find the position of the predicate
            pred_end = line.index('@')
            # Look for the end of the predicate (whitespace)
            while pred_end < len(line) and not line[pred_end].isspace():
                pred_end += 1
            
            pred_text = line[:pred_end].strip()
            line = line[pred_end:].strip()
            
            predicate = PTXOperand(text=pred_text, is_predicate=True)
        
        # Split instruction into opcode and operands
        parts = line.split(None, 1)
        if not parts:
            return  # Empty line
        
        opcode = parts[0]
        operand_text = parts[1] if len(parts) > 1 else ""
        
        # Parse type suffix (e.g., add.s32)
        type_suffix = None
        opcode_parts = opcode.split('.')
        if len(opcode_parts) > 1:
            try:
                # See if the last part is a type
                potential_type = opcode_parts[-1]
                type_suffix = PTXType.from_string(potential_type)
                # If successful, remove type from opcode
                opcode = '.'.join(opcode_parts[:-1])
            except ValueError:
                # Not a type, treat as part of the opcode
                pass
        
        # Parse operands
        operands = []
        if operand_text:
            # Split by commas, but respect parentheses
            in_paren = 0
            start = 0
            for i, c in enumerate(operand_text):
                if c == '(':
                    in_paren += 1
                elif c == ')':
                    in_paren -= 1
                elif c == ',' and in_paren == 0:
                    operand = operand_text[start:i].strip()
                    if operand:
                        operands.append(PTXOperand.from_text(operand))
                    start = i + 1
            
            # Add final operand
            operand = operand_text[start:].strip()
            if operand:
                operands.append(PTXOperand.from_text(operand))
        
        # Create instruction
        instruction = PTXInstruction(
            opcode=opcode,
            type_suffix=type_suffix,
            operands=operands,
            predicate=predicate,
            line_number=self.line_number,
            original_text=line
        )
        
        # Add to current function
        self.current_function.add_instruction(instruction)