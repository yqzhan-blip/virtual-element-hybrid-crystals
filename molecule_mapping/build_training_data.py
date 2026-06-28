# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Training data builder for organic-inorganic hybrid crystals with virtual elements.

Converts full-atom hybrid structures (or generates synthetic ones) into
coarse-grained virtual-element crystals suitable for MatterGen training.

Output format follows CrystalDataset cache convention:
    cache/
    ├── pos.npy
    ├── cell.npy
    ├── atomic_numbers.npy
    ├── num_atoms.npy
    ├── structure_id.npy
    └── chemical_system.json  (optional property)

Usage:
    python build_training_data.py --input-dir ./cif_files --output-dir ./datasets/hybrid_virtual/train
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, PeriodicSite, Element, DummySpecies, Structure
from pymatgen.io.cif import CifParser
from tqdm.auto import tqdm

from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM,
    VIRTUAL_ELEMENT_OFFSET,
    N_VIRTUAL_ELEMENTS,
    is_virtual_element,
    get_virtual_element_z,
    get_virtual_symbol,
)
from mattergen.molecule_mapping.library import (
    VirtualElementLibrary,
    build_quick_perovskite_library,
    build_virtual_element_library,
)
from mattergen.molecule_mapping.cation_library import get_full_cation_library
from mattergen.molecule_mapping.data_convert import (
    identify_organic_fragments,
    compute_centroid,
    convert_hybrid_to_virtual_crystal,
)
from mattergen.molecule_mapping.descriptors import compute_descriptor_from_smiles


# ============================================================================
# Template structures for synthetic data generation
# ============================================================================

# Known 3D perovskite lattice parameters (cubic approximations)
PEROVSKITE_TEMPLATES = {
    "MAPbI3": {
        "lattice": [6.31, 6.31, 6.31],  # cubic, Å
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "I",
        "A_smiles": "C[NH3+]",
        "space_group": "Pm-3m",
    },
    "MAPbBr3": {
        "lattice": [5.93, 5.93, 5.93],
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "Br",
        "A_smiles": "C[NH3+]",
        "space_group": "Pm-3m",
    },
    "MAPbCl3": {
        "lattice": [5.69, 5.69, 5.69],
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "Cl",
        "A_smiles": "C[NH3+]",
        "space_group": "Pm-3m",
    },
    "FAPbI3": {
        "lattice": [6.36, 6.36, 6.36],
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "I",
        "A_smiles": "C(=[NH2+])N",
        "space_group": "Pm-3m",
    },
    "CsPbI3": {
        "lattice": [6.16, 6.16, 6.16],
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "I",
        "A_smiles": None,  # Cs is real element
        "space_group": "Pm-3m",
    },
    "MAPbI3_tetr": {
        "lattice": [8.86, 8.86, 12.68],  # tetragonal
        "angles": [90, 90, 90],
        "B_site": "Pb",
        "X_site": "I",
        "A_smiles": "C[NH3+]",
        "space_group": "I4/mcm",
    },
}

# Tolerance factor heuristics for lattice expansion
# Larger A-site cation -> larger lattice parameter
RADIUS_TO_LATTICE_SCALE = {
    "3D_cubic": 2.8,  # slope: delta_a / delta_r_eff
    "3D_tetr": 2.5,
}


def estimate_lattice_parameter(
    template_name: str,
    new_cation_smiles: str,
    base_lattice: List[float],
) -> List[float]:
    """
    Estimate lattice parameters for a new A-site cation based on template.

    Uses effective radius to scale lattice: larger cation = larger cell.
    """
    template = PEROVSKITE_TEMPLATES[template_name]
    base_smiles = template["A_smiles"]

    # Compute effective radius of new cation
    desc_new = compute_descriptor_from_smiles(new_cation_smiles, charge=1, role="A-site cation")
    r_new = desc_new.r_eff

    if base_smiles is not None:
        desc_base = compute_descriptor_from_smiles(base_smiles, charge=1, role="A-site cation")
        r_base = desc_base.r_eff
    else:
        # Cs template, r_eff ~ 1.67 (ionic radius)
        r_base = 1.67

    dr = r_new - r_base

    # Scale lattice parameters
    scale_factor = 1.0 + dr / 10.0  # heuristic: 1Å radius diff ~ 10% lattice change
    scale_factor = max(0.9, min(1.5, scale_factor))  # clamp

    new_lattice = [a * scale_factor for a in base_lattice]
    return new_lattice


def build_synthetic_perovskite(
    template_name: str,
    a_cation_smiles: str,
    b_site: Optional[str] = None,
    x_site: Optional[str] = None,
    virtual_library: Optional[VirtualElementLibrary] = None,
) -> Tuple[Structure, int]:
    """
    Build a synthetic perovskite structure with given A-site cation.

    Args:
        template_name: Key in PEROVSKITE_TEMPLATES
        a_cation_smiles: SMILES of A-site organic cation
        b_site: Override B-site element (default from template)
        x_site: Override X-site element (default from template)
        virtual_library: For mapping A-site to virtual element

    Returns:
        (virtual_element_structure, virtual_z_of_a_site)
    """
    template = PEROVSKITE_TEMPLATES[template_name]

    B = b_site or template["B_site"]
    X = x_site or template["X_site"]

    # Estimate lattice parameters
    lattice_params = estimate_lattice_parameter(template_name, a_cation_smiles, template["lattice"])
    lattice = Lattice.from_parameters(
        a=lattice_params[0],
        b=lattice_params[1],
        c=lattice_params[2],
        alpha=template["angles"][0],
        beta=template["angles"][1],
        gamma=template["angles"][2],
    )

    # Build cubic perovskite: A at (0,0,0), B at (0.5,0.5,0.5), X at (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # For virtual element representation, A-site is a single virtual atom at the A-site position

    # Map A-site cation to virtual element
    if virtual_library is None:
        virtual_library = build_quick_perovskite_library()

    virtual_z = virtual_library.get_virtual_z(a_cation_smiles, charge=1, role="A-site cation")

    # Sites in fractional coordinates
    sites = []

    # A-site (virtual element)
    sites.append(PeriodicSite(
        species=DummySpecies(get_virtual_symbol(virtual_z), oxidation_state=0),
        coords=[0.0, 0.0, 0.0],
        lattice=lattice,
    ))

    # B-site (real element, e.g., Pb)
    sites.append(PeriodicSite(
        species=Element(B),
        coords=[0.5, 0.5, 0.5],
        lattice=lattice,
    ))

    # X-sites (3 per formula unit, e.g., I)
    for x_pos in [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]:
        sites.append(PeriodicSite(
            species=Element(X),
            coords=list(x_pos),
            lattice=lattice,
        ))

    structure = Structure.from_sites(sites)
    return structure, virtual_z


# ============================================================================
# Structure conversion: full-atom hybrid -> virtual-element crystal
# ============================================================================

def convert_structure_to_virtual_dataset_format(
    structure: Structure,
    virtual_library: VirtualElementLibrary,
    structure_id: str,
    organic_smiles_map: Optional[Dict[int, dict]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert a single hybrid structure to virtual-element crystal format.

    Args:
        structure: Full-atom hybrid structure (pymatgen Structure)
        virtual_library: VirtualElementLibrary for mapping organic fragments
        structure_id: Unique identifier for this structure
        organic_smiles_map: Optional mapping of fragment index -> {"smiles": ..., "charge": ..., "role": ...}

    Returns:
        Dictionary with keys: pos, cell, atomic_numbers, num_atoms, structure_id
        or None if conversion fails.
    """
    try:
        # Convert to virtual-element crystal
        virtual_structure = convert_hybrid_to_virtual_crystal(
            hybrid_structure=structure,
            virtual_library=virtual_library,
            organic_smiles_map=organic_smiles_map,
        )

        # Extract numpy arrays
        frac_coords = virtual_structure.frac_coords.astype(np.float32)
        cell = virtual_structure.lattice.matrix.astype(np.float32)
        # Extract Z: handle DummySpecies (virtual) vs real Element
        def _get_z(site):
            s = str(site.specie.symbol)
            if s.startswith("X") and len(s) > 1 and s[1:].isdigit():
                return int(s[1:])
            return site.specie.Z

        atomic_numbers = np.array([_get_z(site) for site in virtual_structure], dtype=np.int64)
        num_atoms = len(virtual_structure)

        return {
            "pos": frac_coords,
            "cell": cell,
            "atomic_numbers": atomic_numbers,
            "num_atoms": num_atoms,
            "structure_id": structure_id,
        }
    except Exception as e:
        print(f"[WARN] Failed to convert structure {structure_id}: {e}")
        return None


# ============================================================================
# Dataset builder
# ============================================================================

class HybridTrainingDataBuilder:
    """
    Builds training datasets for hybrid perovskite generation with virtual elements.

    Supports three input modes:
    1. CIF files: Read existing hybrid structures from CIF
    2. Synthetic generation: Generate perovskite variants from templates
    3. CSV input: Batch process with specified cations
    """

    def __init__(
        self,
        virtual_library: Optional[VirtualElementLibrary] = None,
        output_dir: str = "./datasets/hybrid_virtual",
    ):
        self.virtual_library = virtual_library or build_quick_perovskite_library()
        self.output_dir = Path(output_dir)
        self.structures: List[Dict[str, np.ndarray]] = []

    def add_from_cif(
        self,
        cif_path: str,
        structure_id: Optional[str] = None,
        organic_smiles_map: Optional[Dict[int, dict]] = None,
    ) -> bool:
        """Add a single CIF file to the dataset."""
        try:
            parser = CifParser(cif_path)
            structures = parser.parse_structures(primitive=True)
            if not structures:
                print(f"[WARN] No structures parsed from {cif_path}")
                return False

            structure = structures[0]
            sid = structure_id or Path(cif_path).stem

            result = convert_structure_to_virtual_dataset_format(
                structure=structure,
                virtual_library=self.virtual_library,
                structure_id=sid,
                organic_smiles_map=organic_smiles_map,
            )
            if result is not None:
                self.structures.append(result)
                return True
            return False
        except Exception as e:
            print(f"[WARN] Error reading {cif_path}: {e}")
            return False

    def add_from_cif_directory(
        self,
        cif_dir: str,
        organic_smiles_map: Optional[Dict[str, Dict[int, dict]]] = None,
    ) -> int:
        """
        Add all CIF files from a directory.

        Args:
            cif_dir: Directory containing .cif files
            organic_smiles_map: Optional dict mapping filename -> fragment mapping

        Returns:
            Number of successfully added structures
        """
        cif_files = sorted(Path(cif_dir).glob("*.cif"))
        if not cif_files:
            print(f"[WARN] No CIF files found in {cif_dir}")
            return 0

        success = 0
        for cif_file in tqdm(cif_files, desc="Processing CIF files"):
            smap = organic_smiles_map.get(cif_file.stem, None) if organic_smiles_map else None
            if self.add_from_cif(str(cif_file), organic_smiles_map=smap):
                success += 1

        print(f"[INFO] Added {success}/{len(cif_files)} structures from {cif_dir}")
        return success

    def add_synthetic_perovskites(
        self,
        cation_smiles_list: List[str],
        templates: Optional[List[str]] = None,
        b_sites: Optional[List[str]] = None,
        x_sites: Optional[List[str]] = None,
    ) -> int:
        """
        Generate synthetic perovskite structures by varying A-site cations.

        Args:
            cation_smiles_list: List of A-site cation SMILES
            templates: List of template names to use (default: all 3D cubic)
            b_sites: List of B-site elements to try (default: ["Pb"])
            x_sites: List of X-site elements to try (default: ["I", "Br", "Cl"])

        Returns:
            Number of generated structures
        """
        templates = templates or ["MAPbI3", "MAPbBr3", "MAPbCl3", "FAPbI3", "CsPbI3"]
        b_sites = b_sites or ["Pb"]
        x_sites = x_sites or ["I", "Br", "Cl"]

        count = 0
        total = len(cation_smiles_list) * len(templates) * len(b_sites) * len(x_sites)

        for cation_smi in tqdm(cation_smiles_list, desc="Generating synthetic perovskites"):
            for template_name in templates:
                for B in b_sites:
                    for X in x_sites:
                        try:
                            structure, vz = build_synthetic_perovskite(
                                template_name=template_name,
                                a_cation_smiles=cation_smi,
                                b_site=B,
                                x_site=X,
                                virtual_library=self.virtual_library,
                            )

                            # Generate unique structure ID
                            hash_input = f"{template_name}_{cation_smi}_{B}_{X}"
                            sid = hashlib.md5(hash_input.encode()).hexdigest()[:12]
                            sid = f"syn_{template_name}_{B}{X}_V{vz}_{sid}"

                            frac_coords = structure.frac_coords.astype(np.float32)
                            cell = structure.lattice.matrix.astype(np.float32)
                            # Extract Z: handle DummySpecies (virtual elements) vs real Element
                            def _get_z(site):
                                s = str(site.specie.symbol)
                                if s.startswith("X") and len(s) > 1 and s[1:].isdigit():
                                    return int(s[1:])
                                return site.specie.Z

                            atomic_numbers = np.array(
                                [_get_z(site) for site in structure], dtype=np.int64
                            )
                            num_atoms = len(structure)

                            self.structures.append({
                                "pos": frac_coords,
                                "cell": cell,
                                "atomic_numbers": atomic_numbers,
                                "num_atoms": num_atoms,
                                "structure_id": sid,
                            })
                            count += 1
                        except Exception as e:
                            print(f"[WARN] Failed to generate {template_name}/{cation_smi}/{B}/{X}: {e}")

        print(f"[INFO] Generated {count}/{total} synthetic structures")
        return count

    def add_from_csv(self, csv_path: str, cif_column: str = "cif") -> int:
        """
        Add structures from a CSV file containing CIF strings.

        Args:
            csv_path: Path to CSV file
            cif_column: Column name containing CIF data

        Returns:
            Number of successfully added structures
        """
        df = pd.read_csv(csv_path)

        if cif_column not in df.columns:
            print(f"[WARN] Column '{cif_column}' not found in {csv_path}")
            return 0

        success = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
            try:
                cif_str = row[cif_column]
                parser = CifParser.from_str(cif_str)
                structures = parser.parse_structures(primitive=True)
                if not structures:
                    continue

                structure = structures[0]
                sid = row.get("material_id", f"csv_{idx}")

                result = convert_structure_to_virtual_dataset_format(
                    structure=structure,
                    virtual_library=self.virtual_library,
                    structure_id=str(sid),
                )
                if result is not None:
                    self.structures.append(result)
                    success += 1
            except Exception as e:
                print(f"[WARN] Error processing row {idx}: {e}")

        print(f"[INFO] Added {success}/{len(df)} structures from CSV")
        return success

    def build_cache(self, split: str = "train") -> str:
        """
        Build and save the dataset cache.

        Args:
            split: Dataset split name (train/val/test)

        Returns:
            Path to the cache directory
        """
        if not self.structures:
            raise ValueError("No structures to build dataset. Add structures first.")

        cache_dir = self.output_dir / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate all structures
        all_pos = np.row_stack([s["pos"] for s in self.structures])
        all_cell = np.array([s["cell"] for s in self.structures])
        all_atomic_numbers = np.concatenate([s["atomic_numbers"] for s in self.structures])
        all_num_atoms = np.array([s["num_atoms"] for s in self.structures], dtype=np.int64)
        # Use integer structure IDs (compatible with np.load without pickle)
        sid_to_idx = {}
        sid_int_list = []
        for sid_str in [s["structure_id"] for s in self.structures]:
            if sid_str not in sid_to_idx:
                sid_to_idx[sid_str] = len(sid_to_idx)
            sid_int_list.append(sid_to_idx[sid_str])
        all_structure_id = np.array(sid_int_list, dtype=np.int64)

        # Save numpy arrays
        np.save(cache_dir / "pos.npy", all_pos)
        np.save(cache_dir / "cell.npy", all_cell)
        np.save(cache_dir / "atomic_numbers.npy", all_atomic_numbers)
        np.save(cache_dir / "num_atoms.npy", all_num_atoms)
        np.save(cache_dir / "structure_id.npy", all_structure_id)

        # Save sid mapping (human-readable)
        sid_map = {str(k): v for k, v in sid_to_idx.items()}
        with open(cache_dir / "structure_id_map.json", "w") as f:
            json.dump(sid_map, f, indent=2)

        # Save metadata
        metadata = {
            "n_structures": len(self.structures),
            "n_atoms_total": int(all_pos.shape[0]),
            "split": split,
            "virtual_library_n_virtual": self.virtual_library.n_virtual,
            "virtual_element_offset": VIRTUAL_ELEMENT_OFFSET,
        }
        with open(cache_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[INFO] Saved cache to {cache_dir}")
        print(f"       Structures: {metadata['n_structures']}")
        print(f"       Total atoms: {metadata['n_atoms_total']}")

        return str(cache_dir)

    def get_statistics(self) -> dict:
        """Get statistics about the collected structures."""
        if not self.structures:
            return {}

        all_atomic_numbers = np.concatenate([s["atomic_numbers"] for s in self.structures])
        real_mask = all_atomic_numbers <= MAX_ATOMIC_NUM
        virtual_mask = (all_atomic_numbers >= VIRTUAL_ELEMENT_OFFSET) & (
            all_atomic_numbers < VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS
        )

        return {
            "n_structures": len(self.structures),
            "total_atoms": len(all_atomic_numbers),
            "real_atoms": int(real_mask.sum()),
            "virtual_atoms": int(virtual_mask.sum()),
            "unique_real_elements": sorted(set(all_atomic_numbers[real_mask].tolist())),
            "unique_virtual_elements": sorted(set(all_atomic_numbers[virtual_mask].tolist())),
            "avg_atoms_per_structure": float(len(all_atomic_numbers) / len(self.structures)),
        }


# ============================================================================
# Quick synthetic dataset generation
# ============================================================================

def build_default_synthetic_dataset(
    output_dir: str = "./datasets/hybrid_virtual",
    n_cations: int = 50,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, str]:
    """
    Build a default synthetic training dataset from known perovskite cations.

    Args:
        output_dir: Output directory for dataset cache
        n_cations: Number of cations to use (from the full cation library)
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation

    Returns:
        Dictionary mapping split names to cache paths
    """
    # Get cation library
    cation_lib = get_full_cation_library()
    all_cations = list(cation_lib.items())

    # Shuffle and select
    np.random.seed(42)
    indices = np.random.permutation(len(all_cations))
    selected = [all_cations[i] for i in indices[:min(n_cations, len(all_cations))]]

    cation_smiles = [info["smiles"] for _, info in selected]
    print(f"[INFO] Selected {len(cation_smiles)} cations for synthetic dataset")

    # Generate all synthetic structures
    builder = HybridTrainingDataBuilder(output_dir=output_dir)
    builder.add_synthetic_perovskites(
        cation_smiles_list=cation_smiles,
        templates=["MAPbI3", "MAPbBr3", "MAPbCl3", "FAPbI3", "CsPbI3"],
        b_sites=["Pb", "Sn"],
        x_sites=["I", "Br", "Cl"],
    )

    # Shuffle
    np.random.shuffle(builder.structures)

    # Split
    n = len(builder.structures)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": builder.structures[:n_train],
        "val": builder.structures[n_train:n_train + n_val],
        "test": builder.structures[n_train + n_val:],
    }

    paths = {}
    for split_name, split_structures in splits.items():
        if not split_structures:
            continue
        split_builder = HybridTrainingDataBuilder(
            virtual_library=builder.virtual_library,
            output_dir=output_dir,
        )
        split_builder.structures = split_structures
        path = split_builder.build_cache(split=split_name)
        paths[split_name] = path

    # Print stats
    print("\n[INFO] Dataset statistics:")
    for split_name, split_structures in splits.items():
        if split_structures:
            all_z = np.concatenate([s["atomic_numbers"] for s in split_structures])
            n_real = int((all_z <= MAX_ATOMIC_NUM).sum())
            n_virt = int((all_z >= VIRTUAL_ELEMENT_OFFSET).sum())
            print(f"  {split_name}: {len(split_structures)} structures, "
                  f"{n_real} real atoms, {n_virt} virtual atoms")

    return paths


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build training data for hybrid perovskite generation with virtual elements"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "cif", "csv"],
        default="synthetic",
        help="Data source mode",
    )
    parser.add_argument("--output-dir", default="./datasets/hybrid_virtual", help="Output cache directory")
    parser.add_argument("--input-dir", help="Input directory (for CIF mode)")
    parser.add_argument("--input-csv", help="Input CSV file (for CSV mode)")
    parser.add_argument("--n-cations", type=int, default=50, help="Number of cations (synthetic mode)")
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--build-all-splits", action="store_true", help="Build train/val/test splits")

    args = parser.parse_args()

    if args.mode == "synthetic":
        if args.build_all_splits:
            build_default_synthetic_dataset(
                output_dir=args.output_dir,
                n_cations=args.n_cations,
            )
        else:
            builder = HybridTrainingDataBuilder(output_dir=args.output_dir)
            cation_lib = get_full_cation_library()
            cation_smiles = [info["smiles"] for info in list(cation_lib.values())[:args.n_cations]]
            builder.add_synthetic_perovskites(cation_smiles_list=cation_smiles)
            builder.build_cache(split=args.split)
            stats = builder.get_statistics()
            print(f"\n[INFO] Statistics: {json.dumps(stats, indent=2)}")

    elif args.mode == "cif":
        if not args.input_dir:
            raise ValueError("--input-dir is required for CIF mode")
        builder = HybridTrainingDataBuilder(output_dir=args.output_dir)
        builder.add_from_cif_directory(args.input_dir)
        builder.build_cache(split=args.split)

    elif args.mode == "csv":
        if not args.input_csv:
            raise ValueError("--input-csv is required for CSV mode")
        builder = HybridTrainingDataBuilder(output_dir=args.output_dir)
        builder.add_from_csv(args.input_csv)
        builder.build_cache(split=args.split)


if __name__ == "__main__":
    main()
