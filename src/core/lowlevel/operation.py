"""
Operation graph representation.

This module defines the core representation of computational operations
that can be compiled using low-level backends.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple


class OperationType(Enum):
    """Enumeration of supported operation types."""
    # Basic arithmetic operations
    ADD = 'add'
    SUBTRACT = 'subtract'
    MULTIPLY = 'multiply'
    DIVIDE = 'divide'
    
    # Vector operations
    DOT_PRODUCT = 'dot_product'
    MATRIX_MULTIPLY = 'matrix_multiply'
    VECTOR_ADD = 'vector_add'
    
    # Memory operations
    LOAD = 'load'
    STORE = 'store'
    
    # Control flow
    LOOP = 'loop'
    CONDITION = 'condition'
    
    # Special operations
    CUSTOM = 'custom'


class DataType(Enum):
    """Enumeration of supported data types."""
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'
    FLOAT16 = 'float16'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    COMPLEX64 = 'complex64'
    COMPLEX128 = 'complex128'
    BOOL = 'bool'
    
    # Special data types
    BF16 = 'bfloat16'  # Brain floating point (used in ML)
    FP8 = 'float8'     # 8-bit floating point (used in ML)


class Operation:
    """Representation of a computational operation."""
    
    def __init__(self, 
                 op_type: Union[OperationType, str],
                 name: str,
                 inputs: List[Union['Operation', str]],
                 output_type: Union[DataType, str],
                 attributes: Optional[Dict[str, Any]] = None):
        """
        Initialize operation.
        
        Args:
            op_type: Operation type
            name: Operation name (used as identifier)
            inputs: Input operations or variable names
            output_type: Data type of operation output
            attributes: Additional operation attributes
        """
        # Convert string to enum if needed
        if isinstance(op_type, str):
            try:
                op_type = OperationType(op_type)
            except ValueError:
                op_type = OperationType.CUSTOM
                
        if isinstance(output_type, str):
            try:
                output_type = DataType(output_type)
            except ValueError:
                raise ValueError(f"Unknown data type: {output_type}")
        
        self.op_type = op_type
        self.name = name
        self.inputs = inputs
        self.output_type = output_type
        self.attributes = attributes or {}
        
    def __repr__(self) -> str:
        """Get string representation."""
        return f"Operation({self.op_type.value}, name={self.name}, inputs={[i.name if isinstance(i, Operation) else i for i in self.inputs]})"


class OperationGraph:
    """Graph of computational operations."""
    
    def __init__(self, 
                 operations: List[Operation],
                 inputs: List[Tuple[str, DataType]],
                 outputs: List[str],
                 name: str = "compute_graph"):
        """
        Initialize operation graph.
        
        Args:
            operations: List of operations in the graph
            inputs: List of input variable names and types
            outputs: List of output variable names
            name: Name of the graph
        """
        self.operations = operations
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        
        # Build operation map for quick lookup
        self.operation_map = {op.name: op for op in operations}
        
    def get_operation(self, name: str) -> Optional[Operation]:
        """
        Get operation by name.
        
        Args:
            name: Operation name
            
        Returns:
            Operation object or None if not found
        """
        return self.operation_map.get(name)
    
    def get_input_operations(self) -> List[Operation]:
        """
        Get operations that depend directly on input variables.
        
        Returns:
            List of operations
        """
        input_vars = [name for name, _ in self.inputs]
        return [op for op in self.operations 
                if any(isinstance(inp, str) and inp in input_vars for inp in op.inputs)]
    
    def get_output_operations(self) -> List[Operation]:
        """
        Get operations that produce output variables.
        
        Returns:
            List of operations
        """
        return [op for op in self.operations if op.name in self.outputs]
    
    def __repr__(self) -> str:
        """Get string representation."""
        return f"OperationGraph(name={self.name}, operations={len(self.operations)}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"