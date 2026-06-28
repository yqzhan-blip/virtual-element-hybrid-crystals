# GitHub Upload Guide

## Repository Name

**Recommended**: `virtual-element-hybrid-crystals`

### Alternative options
- `mattergen-hybrid-crystals` — emphasizes the MatterGen foundation
- `ve-hybrid-crystals` — shorter, uses virtual-element acronym
- `hybrid-crystal-generation` — broader, less specific to the method
- `virtual-element-framework` — highlights the methodology over the application

### Naming rationale
The recommended name is lowercase, hyphen-separated, descriptive, and captures the two key ideas: **virtual-element representation** + **organic–inorganic hybrid crystals**. It is also unique enough to avoid collisions while remaining search-friendly.

## Repository Description (350 chars max)

A virtual-element framework extending pretrained inorganic crystal diffusion models (MatterGen) to organic–inorganic hybrid materials. Maps organic cations to coarse-grained pseudo-elements, fine-tunes generation, decodes full-atom structures, and validates with DFT. Code, figures, and manuscript for the paper submission.

(347 characters)

## What to Upload

### 1. Core virtual-element framework
- `mattergen/mattergen/molecule_mapping/`
  - `virtual_elements.py` — virtual element constants and gap masking
  - `descriptors.py` — molecular descriptor calculations
  - `library.py` — virtual element library (K-means mapping)
  - `data_convert.py` — hybrid ↔ virtual-element conversion
  - `extended_embedding.py` — extended atom embedding layer
  - `extended_chemical_system.py` — extended chemical system embedding
  - `decoder.py` — full-atom reconstruction from virtual elements
  - `ion_classes.py` — 12 coarse-grained ion functional classes
  - `cation_library.py` — organic cation template library
  - `__init__.py`

### 2. Dataset construction
- `mattergen/mattergen/molecule_mapping/build_training_data.py`
- `mattergen/mattergen/molecule_mapping/build_amplified_dataset.py`
- `mattergen/mattergen/molecule_mapping/build_expanded_dataset.py`
- `mattergen/analyze_balanced.py`
- `mattergen/analyze_perovskite_candidates.py`

### 3. Generation, decoding, and validation pipeline
- `run_paper_pipeline.py`
- `check_occ.py`
- `clean_pipeline.py`
- `convert_to_qe.py` / `convert_to_qe_lowmem.py`
- `build_3d_templates.py`

### 4. DFT extraction and analysis
- `paper_results/dft_calculator.py`
- `extract_band_structure.py`
- `extract_corrected_gap_v2.py`
- `extract_dft_*.py`
- `extract_electronic_structure.py`
- `extract_fermi_gap.py`
- `extract_final_gap.py`
- `extract_gap*.py`
- `extract_halides.py`
- `extract_homo_lumo_gap.py`
- `plot_all_bands.py`

### 5. Figures and manuscript
- `paper_results/generate_nature_figures.py`
- `paper_results/generate_paper_docx_v2.py`
- `paper_results/generate_si_docx.py`
- `paper_results/docx_utils.py`
- `paper_results/figures_nature/` (Nature-style Figure 1–5)
- `paper_results/paper_draft_v5.md`
- `paper_results/SI_v5.md`
- `paper_results/README.md`

### 6. Key data and configs
- `paper_results/configs/`
- `paper_results/dft_inputs/` (input templates only, exclude large pseudopotential files)
- `paper_results/dft_*.json`
- `paper_results/50_halide_structures.csv`
- `paper_results/dft_selection.json`
- `paper_results/validation_report.json`
- Selected sample decoded CIFs in `paper_results/decoded_cifs/` or `mattergen/decoded_crystals_v9/`

### 7. Documentation
- `README.md` (create a top-level repo README summarizing installation, usage, and citation)
- `GITHUB_UPLOAD_GUIDE.md` (this file)
- `.gitignore`
- `requirements.txt` or `environment.yml`
- `LICENSE` (choose an open-source license)

## What NOT to Upload

- **Large model checkpoints** (>100 MB). Use Git LFS, Hugging Face, or a separate data repository.
- **Massive raw CIF collections** from intermediate generations. Keep only representative samples.
- **DFT pseudopotential files** (e.g., SSSP UPF files) — these are large and publicly downloadable.
- **Large DFT output files** (`.scf.out`, charge density files, WFC files).
- **Full DFT input/output directories** (e.g., `dft_inputs/crystal_*_band/` with `.save/` folders) — only input templates are included.
- **`.workbuddy/`**, `__pycache__/`, `.pipcache/`, `.pytest_cache/`.
- **Large zip archives**: `mattergen-main.zip`, `stoned-selfies-main.zip`, `paper_results.zip`.
- **Personal/memory logs** in `.workbuddy/memory/`.
- **Credentials, API keys, or private tokens**.
- **Binary executables** (e.g., `pw.exe` paths in configs should be local-only).

## Local Repository Prepared for Push

A local git repository has been prepared at:

```
D:/virtual-element-hybrid-crystals/
```

It is already initialized and committed. To push to GitHub, see the commands below.

## Push to GitHub

### Option A: Let me push automatically

Provide your GitHub username and a personal access token (PAT) with `repo` scope.

### Option B: Push manually

1. Create the repository on GitHub (web interface).
2. Run:

```bash
cd D:/virtual-element-hybrid-crystals
git branch -m main
git remote add origin https://github.com/YOUR_USERNAME/virtual-element-hybrid-crystals.git
# For HTTPS with PAT:
# git remote add origin https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/virtual-element-hybrid-crystals.git
git push -u origin main
```

## Suggested Top-Level Repository Structure

```
virtual-element-hybrid-crystals/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── molecule_mapping/          # Core framework (from mattergen/mattergen/molecule_mapping/)
│   ├── __init__.py
│   ├── virtual_elements.py
│   ├── descriptors.py
│   ├── library.py
│   ├── data_convert.py
│   ├── extended_embedding.py
│   ├── extended_chemical_system.py
│   ├── decoder.py
│   ├── ion_classes.py
│   ├── cation_library.py
│   ├── build_training_data.py
│   ├── build_amplified_dataset.py
│   └── build_expanded_dataset.py
├── pipelines/                 # Generation & DFT workflows
│   ├── run_paper_pipeline.py
│   ├── check_occ.py
│   ├── convert_to_qe.py
│   ├── dft_calculator.py
│   └── extract_*.py
├── paper/                     # Manuscript & figures
│   ├── generate_nature_figures.py
│   ├── generate_paper_docx_v2.py
│   ├── generate_si_docx.py
│   ├── docx_utils.py
│   ├── paper_draft_v5.md
│   ├── SI_v5.md
│   └── figures/
├── data/                      # Small metadata & sample structures
│   ├── configs/
│   ├── dft_inputs/
│   ├── dft_*.json
│   ├── 50_halide_structures.csv
│   └── sample_decoded_cifs/
└── analysis/                  # Analysis scripts
    ├── analyze_balanced.py
    ├── analyze_perovskite_candidates.py
    └── plot_all_bands.py
```

## Quick Setup Checklist

1. Create a new GitHub repository.
2. Copy the files listed above into the suggested structure.
3. Write a top-level `README.md` with installation instructions.
4. Add a `.gitignore` file (see below).
5. Add a `LICENSE` file.
6. Test that `import molecule_mapping` works in a fresh environment.
7. Push to GitHub.

## Example `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
venv/

# WorkBuddy local files
.workbuddy/

# Large archives
*.zip
*.tar.gz
*.tar

# Large data / models
checkpoints/
*.ckpt
*.pth
*.pt

# DFT large outputs
*.scf.out
*.save/
charge-density.dat
wfc*

# Pseudopotentials (large, publicly available)
*.UPF
pseudos/

# Local OS files
.DS_Store
Thumbs.db
```
