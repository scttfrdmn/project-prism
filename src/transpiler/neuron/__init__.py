"""
AWS Neuron IR generation and optimization.

This module provides functionality for generating AWS Neuron IR code
from parsed PTX code, targeting both Trainium and Inferentia hardware.
"""

from .module import NeuronModule, NeuronFunction, NeuronInstruction, NeuronOperand
from .targets import NeuronTarget, TrainiumTarget
from .inferentia import InferentiaTranslator

__all__ = [
    'NeuronModule',
    'NeuronFunction',
    'NeuronInstruction',
    'NeuronOperand',
    'NeuronTarget',
    'TrainiumTarget',
    'InferentiaTranslator',
]