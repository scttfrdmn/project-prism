# Project Prism

A framework for scientific computing on AWS Silicon and other hardware platforms.

## Overview

Prism (Parallel Research Infrastructure for Scientific Multiprocessing) is a framework designed to enable scientific computing across diverse hardware platforms, with specific optimizations for AWS Silicon (Graviton, Trainium, and Inferentia). It provides a unified programming model that allows scientists to write code once and run it efficiently on multiple hardware targets.

## Key Features

- **Multi-Language Support**: C/C++, Python, Fortran, Julia, and Rust interfaces
- **Cross-Platform Execution**: Run the same code on AWS Silicon, NVIDIA GPUs, and FPGA instances
- **Domain-Specific Libraries**: Specialized libraries for genomics, chemistry, climate science, and more
- **Layered API Design**: From high-level domain concepts to low-level hardware control
- **Hardware-Aware Optimization**: Automatic selection of optimal hardware for each workload
- **NVIDIA to AWS Silicon Transpilation**: Automatically convert NVIDIA PTX code to run on AWS Trainium and Inferentia

## Supported Hardware

### AWS Silicon
- **Trainium**: Specialized for machine learning training workloads
  - Trainium 1 (Trn1): First generation AWS ML training chips
  - Trainium 2 (Coming soon): Second generation with improved performance
- **Inferentia**: Designed for machine learning inference
  - Inferentia 1 (Inf1): First generation inference processor
  - Inferentia 2 (Inf2): Second generation with improved performance and wider model support
- **Graviton**: ARM-based processors for general computation
  - Graviton 1: First-generation ARM processor (Cortex-A72 cores)
  - Graviton 2: Second-generation ARM processor (Neoverse N1 cores)
  - Graviton 3: Third-generation with 256-bit SVE, BF16 support (Neoverse V1 cores)
  - Graviton 3E: Enhanced version for networking (C7gn) and HPC (HPC7g) with 35% higher vector performance
  - Graviton 4: Latest generation with 512-bit SVE2, FP8 support, confidential computing
- **FPGA (F1)**: Field Programmable Gate Arrays for custom hardware acceleration
  - F1.2xlarge: Single FPGA instance
  - F1.4xlarge: Dual FPGA instance
  - F1.16xlarge: 8 FPGA instance

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
│   │   ├── kernel.py         # Computational kernel abstraction
│   │   └── capability.py     # Hardware capability detection & selection
│   ├── backends/             # Hardware-specific backends
│   │   ├── trainium.py       # AWS Trainium implementation
│   │   ├── trainium_v1.py    # Trainium 1 specific optimizations  
│   │   ├── trainium_v2.py    # Trainium 2 specific optimizations
│   │   ├── inferentia.py     # AWS Inferentia implementation
│   │   ├── inferentia_v1.py  # Inferentia 1 specific optimizations
│   │   ├── inferentia_v2.py  # Inferentia 2 specific optimizations
│   │   ├── graviton.py       # AWS Graviton implementation with versioning
│   │   └── fpga.py           # AWS F1 FPGA implementation
│   ├── domains/              # Domain-specific libraries
│   │   └── genomics.py       # Genomics functionality
│   ├── transpiler/           # Cross-architecture code conversion
│   │   ├── ptx/              # NVIDIA PTX parsing and representation
│   │   └── neuron/           # AWS Neuron IR generation components
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
import prism.domains.genomics as prg

# Load data
sequences = prg.read_fastq("samples.fastq")
reference = prg.read_fasta("reference.fa")

# Analyze with automatic hardware selection
alignments = prg.align(sequences, reference)
variants = prg.call_variants(alignments)

# Explicitly target specific hardware
with pr.device("aws.trainium"):
    specialized_results = prg.analyze_variants(sequences, reference)

# Target specific hardware versions
with pr.device("aws.inferentia", version=2):
    inf2_results = prg.infer(model, input_data)

# Target specific Graviton versions - general API
with pr.device("aws.graviton", version="3E"):
    # Optimized for enhanced networking on Graviton 3E
    network_results = prg.network_analysis(data)

# Using specialized Graviton context managers - recommended approach
with pr.graviton3e():
    # Automatically optimized for Graviton 3E with enhanced networking
    network_results = prg.network_analysis(data)

with pr.graviton4():
    # Utilize Graviton 4's 512-bit SVE2 and advanced features
    hpc_results = prg.compute_simulation(data)

# Automatic backend selection based on workload
device = pr.create_optimal_device(
    workload_type="inference",
    batch_size=128,
    model_size="large",
    precision="bf16"
)
```

## NVIDIA PTX to AWS Silicon Transpilation

Prism includes a sophisticated transpiler for converting NVIDIA PTX code to AWS Neuron IR:

```python
from prism.transpiler import transpile_ptx_to_neuron

# Convert NVIDIA PTX to Neuron IR for Inferentia 2
neuron_module = transpile_ptx_to_neuron(
    ptx_file="kernel.ptx",
    target="inferentia2",
    optimization_level=2
)

# Save the transpiled module
neuron_module.save("kernel_inferentia.neuron")
```

## Graviton-Specific Optimizations

Prism provides specialized optimizations for different Graviton processors with an easy-to-use API:

```python
import prism as pr

# Using the context manager API (recommended)
with pr.graviton3e() as device:
    # Get optimal vector width for Graviton 3E
    vector_width = device.get_optimal_vector_width()
    print(f"Optimal vector width: {vector_width} bits")  # 256 bits
    
    # Optimize memory alignment 
    alignment = device.get_optimal_memory_alignment()
    print(f"Optimal memory alignment: {alignment} bytes")  # 64 bytes
    
    # Apply HPC-specific optimizations for MPI workloads
    if device.is_hpc_optimized():
        device.optimize_for_hpc(hpc_type="mpi")
        sve_config = device.get_sve_configuration()
        print(f"Using 256-bit SVE with enhanced vector performance (+35%)")
    
    # Run optimized computation
    result = compute_with_device(device, data)

# Using the context manager with Graviton 4
with pr.graviton4() as device:
    # Use SVE2 on Graviton 4 with 512-bit vectors
    sve_config = device.get_sve_configuration()
    print(f"Using SVE2 with {sve_config['vector_length']}-bit vectors")  # 512 bits
    
    # Enable confidential computing features
    if device.enable_confidential_computing():
        print("Confidential computing enabled")
    
    # Benefit from FP8 support for ML workloads
    if device.get_capabilities()["fp8_support"]:
        use_fp8_optimizations = True

# Direct instantiation approach (alternative)
device = pr.backends.Graviton3Device()

# Version-specific features with direct instance
if device.version_str == "3E":
    # Apply Graviton 3E networking optimizations
    if device.is_network_optimized():
        device.optimize_for_networking(network_type="throughput")
        print(f"Using enhanced network capabilities with 200 Gbps bandwidth")
elif int(device.version_str) == 3:
    # Use SVE on Graviton 3
    sve_config = device.get_sve_configuration()
    print(f"Using SVE with vector length: {sve_config['vector_length']} bits")
```

## Advanced Graviton Usage

The framework provides automatic version detection and specialized optimizations:

```python
import prism as pr

# Automatically detect and select optimal Graviton version for workload
device = pr.create_optimal_device(
    workload_type="hpc",
    network_intensive=True
)  # Will select Graviton 3E on HPC7g instances

# Use specific Graviton versions for different workloads
computations = []
data = get_input_data()

# Benchmark across all Graviton versions
for version in [1, 2, 3, "3E", 4]:
    with pr.graviton(version=version) as device:
        # Run computation with version-specific optimizations
        start_time = time.time()
        result = compute_with_device(device, data)
        end_time = time.time()
        
        computations.append({
            "version": version,
            "time": end_time - start_time,
            "result": result
        })

# Compare results and performance
for comp in computations:
    print(f"Graviton {comp['version']}: {comp['time']:.2f} seconds")
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