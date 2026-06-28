# DFT Input Templates

This directory contains example Quantum ESPRESSO input templates. Full DFT input/output files and pseudopotentials are excluded from the repository due to their large size.

## Pseudopotentials

Download pseudopotentials from the SSSP library:
- <https://www.materialscloud.org/discover/sssp>

Place the required `.UPF` files in `./pseudos/` before running calculations.

## Running DFT

See `pipelines/dft_calculator.py` for the automated DFT workflow used in the paper.
