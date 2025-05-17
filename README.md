# Project Prism

A framework for scientific computing on AWS Silicon and other hardware platforms.

## Overview

Prism (Parallel Research Infrastructure for Scientific Multiprocessing) is a framework designed to enable scientific computing across diverse hardware platforms, with specific optimizations for AWS Silicon (Graviton, Trainium, and Inferentia). It provides a unified programming model that allows scientists to write code once and run it efficiently on multiple hardware targets.

## Key Features

- **Multi-Language Support**: C/C++, Python, Fortran, Julia, and Rust interfaces
- **Cross-Platform Execution**: Run the same code on AWS Silicon and NVIDIA GPUs 
- **Domain-Specific Libraries**: Specialized libraries for genomics, chemistry, climate science, and more
- **Layered API Design**: From high-level domain concepts to low-level hardware control
- **Hardware-Aware Optimization**: Automatic selection of optimal hardware for each workload

## Project Structure

```
project-prism/
├── docs/                     # Documentation
│   ├── prism-language-design.md
│   ├── prism-pluggable-backend.md
│   └── prism-practical-usage.md
├── src/                      # Source code
│   ├── core/                 # Core functionality
│   │   ├── device.py         # Device abstraction
│   │   └── kernel.py         # Computational kernel abstraction
│   ├── backends/             # Hardware-specific backends
│   │   └── graviton.py       # AWS Graviton implementation
│   ├── domains/              # Domain-specific libraries
│   │   └── genomics.py       # Genomics functionality
│   └── utils/                # Utility functions
├── tests/                    # Test suite
│   ├── core/                 # Core tests
│   ├── domains/              # Domain tests
│   └── backends/             # Backend tests
├── setup.py                  # Package setup
├── pytest.ini                # Test configuration
└── LICENSE                   # Apache 2.0 license
```

## Installation

```bash
# Not yet available on PyPI
git clone https://github.com/scttfrdmn/project-prism.git
cd project-prism
pip install -e .
```

## Quick Start

```python
import prism as pr
import prism.genomics as prg

# Load data
sequences = prg.read_fastq("samples.fastq")
reference = prg.read_fasta("reference.fa")

# Analyze with automatic hardware selection
alignments = prg.align(sequences, reference)
variants = prg.call_variants(alignments)

# Explicitly target specific hardware
with pr.device(type="aws.trainium"):
    specialized_results = prg.analyze_variants(sequences, reference)
```

## Documentation

For detailed information, see the documentation files in the `docs/` directory:

- [Language Design](docs/prism-language-design.md) - Details on Prism's multi-language ecosystem
- [Pluggable Backend](docs/prism-pluggable-backend.md) - Architecture for supporting multiple hardware platforms
- [Practical Usage](docs/prism-practical-usage.md) - Examples and coding patterns

## Development Status

Prism is currently in early development. The API is subject to change.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

Scott Friedman