# Virtual-Element Hybrid Crystals

A virtual-element framework extending pretrained inorganic crystal diffusion models (MatterGen) to organic–inorganic hybrid materials.

[![Paper](paper/figures/figure_1_pipeline.png)](paper/figures/figure_1_pipeline.png)

## Overview

This repository provides the code, data, and manuscript sources for extending pretrained inorganic crystal generators to organic–inorganic hybrid crystals. The key idea is to map organic cation families to a small set of coarse-grained **virtual elements** (pseudo-atoms with extended atomic numbers), fine-tune the pretrained MatterGen diffusion model on hybrid crystal data, and decode the generated virtual-element crystals back into full-atom hybrid structures.

## Repository Structure

```
virtual-element-hybrid-crystals/
├── molecule_mapping/        # Core virtual-element framework
├── pipelines/               # Generation, decoding, DFT, and analysis workflows
├── paper/                   # Manuscript, figures, and plotting scripts
├── data/                    # Configs, DFT inputs/outputs, sample decoded CIFs
├── analysis/                # Statistics and diversity analysis scripts
├── .gitignore               # Files excluded from version control
├── GITHUB_UPLOAD_GUIDE.md   # Detailed upload checklist
├── LICENSE                  # License
└── README.md                # This file
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/virtual-element-hybrid-crystals.git
cd virtual-element-hybrid-crystals
```

### 2. Create a Python environment

We recommend Python 3.10 or 3.11.

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

- `torch` >= 2.0
- `pymatgen`
- `numpy`, `scipy`, `pandas`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `python-docx` (for manuscript generation)
- `rdkit` (optional; fallback descriptors are provided)

### 4. Install MatterGen base code

This repository extends the original MatterGen model. Install the official MatterGen package or place the MatterGen source in your `PYTHONPATH`:

```bash
pip install mattergen
# OR
git clone https://github.com/microsoft/mattergen.git
export PYTHONPATH=$PYTHONPATH:/path/to/mattergen
```

## Quick Start

### Build the virtual-element library

```python
from molecule_mapping import VirtualElementLibrary, MolecularDescriptor

lib = VirtualElementLibrary(n_clusters=12)
# See molecule_mapping/library.py for full API
```

### Fine-tune MatterGen on hybrid data

```bash
cd pipelines
python run_paper_pipeline.py --config configs/finetune_hybrid.yaml
```

### Decode virtual-element crystals

```python
from molecule_mapping import VirtElem2MolDecoder

decoder = VirtElem2MolDecoder()
structure = decoder.decode(virtual_element_structure)
```

### Reproduce figures

```bash
cd paper
python generate_nature_figures.py
```

### Run DFT validation

```bash
cd pipelines
python dft_calculator.py --config data/configs/dft_config.yaml
```

## Data

- `data/configs/` — training and generation configuration files
- `data/dft_inputs/` — Quantum ESPRESSO input templates
- `data/sample_decoded_cifs/` — representative decoded hybrid crystal structures
- `data/dft_*.json` — DFT electronic-structure analysis results
- `data/50_halide_structures.csv` — charge-neutral halide-containing candidates

## Manuscript

The manuscript draft and supporting information are located in `paper/`:

- `paper/paper_draft_v5.md` — main manuscript
- `paper/SI_v5.md` — supporting information
- `paper/figures/` — Nature-style Figure 1–5

## Citation

If you use this code or data, please cite:

```bibtex
@article{zhan2026virtual,
  title={A virtual-element framework for generative design of organic--inorganic hybrid crystals},
  author={Zhan, Yiqiang},
  journal={npj Computational Materials},
  year={2026}
}
```

## Funding

This work was supported by the Shanghai Science and Technology Innovation Action Plan (No. 24DZ3001200), the National Natural Science Foundation of China (No. 62304046 and No. 62274040), and the Basic Research Project of State Key Laboratory of Photovoltaic Science and Technology (No. 202401020302).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Yiqiang Zhan — zhan@fudan.edu.cn
