#!/usr/bin/env python3
"""
Expanded training dataset builder for organic-inorganic hybrid crystals.
Adds halides / oxides / sulfides with non-perovskite templates.

Covers:
  Halides:     ABX₃ (3D), A₂BX₄ (RP 2D), A₂BX₆ (vacancy DP), A₃B₂X₉
  Oxides:      ABO₃ (perovskite), AB₂O₄ (spinel), A₂B₂O₇ (pyrochlore), A₂BB'O₆ (DP)
  Sulfides:    ABS₃, AB₂S₄ (thiospinel), A₂B₂S₇ (pyrochlore sulfide)
  Others:      A₂BX₆ (elpasolite)

Usage:
  python build_expanded_dataset.py --output-dir datasets/hybrid_expanded --build-all-splits
"""

import hashlib, json, sys, os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Element, DummySpecies, Structure
from tqdm.auto import tqdm

# Ensure mattergen package is importable
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM, VIRTUAL_ELEMENT_OFFSET, N_VIRTUAL_ELEMENTS,
    get_virtual_symbol,
)
from mattergen.molecule_mapping.ion_classes import ION_CLASSES, CATION_TO_CLASS
from mattergen.molecule_mapping.cation_library import get_full_cation_library

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Element and ion candidates
# ═══════════════════════════════════════════════════════════════════════════════

# Metals by oxidation state
METALS = {
    "+1": ["Cu", "Ag", "Li", "Na", "K", "Rb", "Cs", "Tl", "In"],
    "+2": ["Pb", "Sn", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Ge", "Ca", "Sr", "Ba", "Mg", "Cr", "V"],
    "+3": ["Bi", "Sb", "In", "Ga", "Al", "Sc", "Y", "La", "Cr", "Fe", "Mn", "Co", "Ru", "Rh", "Ir"],
    "+4": ["Ti", "Zr", "Hf", "Sn", "Ge", "Mn", "Ce", "Th", "U", "V", "Nb", "Ta", "Mo", "W"],
    "+5": ["Nb", "Ta", "Sb", "Bi", "V", "Mo", "W"],
}

# Anions
ANIONS = {
    "halide": ["F", "Cl", "Br", "I"],
    "oxide":  ["O"],
    "sulfide": ["S", "Se", "Te"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Structure templates (lattice params, site positions, charge formula)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Template:
    """A structure template with charge-balanced composition formula."""
    name: str           # e.g. "ABX₃_perovskite"
    family: str         # "halide" / "oxide" / "sulfide"
    stoichiometry: Dict[str, int]  # e.g. {"A":1, "B":1, "X":3}
    charge_formula: str # e.g. "⊕+2⊕+3⊖" = A⁺ + B²⁺ + 3X⁻
    lattice_scale: float  # typical cubic a in Å
    sites_frac: List[Tuple[float, float, float]]  # fractional coords
    site_labels: List[str]  # "A", "B", "X" matching stoichiometry
    angles: Tuple[float, float, float] = (90, 90, 90)

    @property
    def num_atoms_per_unit(self) -> int:
        return sum(self.stoichiometry.values())


# ── Halide Templates ──────────────────────────────────────────────────────────

HALIDE_TEMPLATES = [
    Template("ABX3_perov", "halide",
             {"A":1, "B":1, "X":3}, "1@1,1@2,3@-1",
             lattice_scale=6.0,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
             site_labels=["A","B","X","X","X"]),

    Template("A2BX4_RP", "halide",
             {"A":2, "B":1, "X":4}, "2@1,1@2,4@-1",
             lattice_scale=6.0,
             sites_frac=[(0,0,0), (0,0,0.5), (0.5,0.5,0.25),
                         (0.5,0,0.25), (0,0.5,0.25),
                         (0.5,0.5,0.75), (0.5,0,0.75), (0,0.5,0.75)],
             site_labels=["A","A","B","X","X","X","X"],
             angles=(90,90,90)),

    Template("A2BX6_vacDP", "halide",
             {"A":2, "B":1, "X":6}, "2@1,1@4,6@-1",
             lattice_scale=10.0,
             sites_frac=[(0.25,0.25,0.25), (0.75,0.75,0.75),
                         (0,0,0),
                         (0.5,0,0), (0,0.5,0), (0,0,0.5),
                         (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
             site_labels=["A","A","B","X","X","X","X","X","X"]),

    Template("A3B2X9", "halide",
             {"A":3, "B":2, "X":9}, "3@1,2@3,9@-1",
             lattice_scale=8.0,
             sites_frac=[(0,0,0), (0.5,0.5,0), (0,0,0.5),
                         (0.33,0.33,0.25), (0.67,0.67,0.75),
                         (0.33,0,0.25), (0.67,0,0.75),
                         (0,0.33,0.25), (0,0.67,0.75),
                         (0.33,0.33,0), (0.67,0.67,0.5),
                         (0.33,0,0), (0.67,0,0.5),
                         (0,0.33,0), (0,0.67,0.5)],
             site_labels=["A","A","A","B","B"] + ["X"]*9),

    # Elpasolite A₂BB'X₆ — good for mixed-valence metals
    Template("A2BBpX6_elp", "halide",
             {"A":2, "B":1, "Bp":1, "X":6}, "2@1,1@2,1@3,6@-1",
             lattice_scale=10.5,
             sites_frac=[(0.25,0.25,0.25), (0.75,0.75,0.75),
                         (0,0,0), (0.5,0.5,0.5),
                         (0.25,0,0), (0.75,0,0),
                         (0,0.25,0), (0,0.75,0),
                         (0,0,0.25), (0,0,0.75)],
             site_labels=["A","A","B","Bp","X","X","X","X","X","X"]),
]

# ── Oxide Templates ───────────────────────────────────────────────────────────

OXIDE_TEMPLATES = [
    Template("ABO3_perov", "oxide",
             {"A":1, "B":1, "X":3}, "1@1,1@2,3@-2",
             lattice_scale=3.9,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
             site_labels=["A","B","O","O","O"]),

    Template("AB2O4_spinel", "oxide",
             {"A":1, "B":2, "X":4}, "1@2,2@3,4@-2",
             lattice_scale=8.3,
             sites_frac=[(0.125,0.125,0.125),  # A (tetrahedral 8a)
                         (0.5,0.5,0.5), (0.5,0.75,0.75),  # B (octahedral 16d)
                         (0.25,0.25,0.25), (0.25,0.25,0.75),
                         (0.25,0.75,0.25), (0.25,0.75,0.75)],
             site_labels=["A","B","B","O","O","O","O"]),

    Template("A2B2O7_pyro", "oxide",
             {"A":2, "B":2, "X":7}, "2@3,2@4,7@-2",
             lattice_scale=10.2,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0.5,0.5,0.5), (0.25,0.75,0.75),
                         (0.125,0.125,0.125), (0.125,0.625,0.625),
                         (0.625,0.125,0.625), (0.625,0.625,0.125),
                         (0.375,0.375,0.375), (0.375,0.875,0.875),
                         (0.875,0.375,0.875)],
             site_labels=["A","A","B","B","O","O","O","O","O","O","O"]),

    Template("A2BBpO6_DP", "oxide",
             {"A":2, "B":1, "Bp":1, "X":6}, "2@1,1@2,1@5,6@-2",
             lattice_scale=7.9,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0,0,0), (0.5,0.5,0.5),
                         (0.25,0,0), (0.75,0,0),
                         (0,0.25,0), (0,0.75,0),
                         (0,0,0.25), (0,0,0.75)],
             site_labels=["A","A","B","Bp","O","O","O","O","O","O"]),
]

# ── Sulfide Templates ─────────────────────────────────────────────────────────

SULFIDE_TEMPLATES = [
    Template("ABS3_perov", "sulfide",
             {"A":1, "B":1, "X":3}, "1@1,1@2,3@-2",
             lattice_scale=5.6,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)],
             site_labels=["A","B","S","S","S"]),

    Template("AB2S4_thio", "sulfide",
             {"A":1, "B":2, "X":4}, "1@2,2@3,4@-2",
             lattice_scale=10.0,
             sites_frac=[(0.125,0.125,0.125),
                         (0.5,0.5,0.5), (0.5,0.75,0.75),
                         (0.25,0.25,0.25), (0.25,0.25,0.75),
                         (0.25,0.75,0.25), (0.25,0.75,0.75)],
             site_labels=["A","B","B","S","S","S","S"]),

    Template("A2B2S7_pyro", "sulfide",
             {"A":2, "B":2, "X":7}, "2@2,1@4,7@-2",
             lattice_scale=10.5,
             sites_frac=[(0,0,0), (0.5,0.5,0.5),
                         (0.5,0.5,0.5), (0.25,0.75,0.75),
                         (0.125,0.125,0.125), (0.125,0.625,0.625),
                         (0.625,0.125,0.625), (0.625,0.625,0.125),
                         (0.375,0.375,0.375), (0.375,0.875,0.875),
                         (0.875,0.375,0.875)],
             site_labels=["A","A","B","B","S","S","S","S","S","S","S"]),
]

ALL_TEMPLATES = HALIDE_TEMPLATES + OXIDE_TEMPLATES + SULFIDE_TEMPLATES

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Charge-balanced composition enumerator
# ═══════════════════════════════════════════════════════════════════════════════

def parse_charge_formula(cf: str) -> list:
    """Parse '1@2,3@-1' → [(1, +2), (3, -1)]"""
    charges = []
    for part in cf.split(","):
        count_str, charge_str = part.split("@")
        charges.append((int(count_str), int(charge_str)))
    return charges


def enumerate_charge_balanced_compositions(
    template: Template,
    n_virtual_cations: int = 5,
    n_metal_combos: int = 20,
) -> List[Dict[str, any]]:
    """
    Enumerate charge-balanced metal+anion combinations for a template.
    Returns list of {virtual_class_Z, metals, anions, formula_str}
    """
    charge_spec = parse_charge_formula(template.charge_formula)
    # charge_spec: [(count_A, q_A), (count_B, q_B), (count_X, q_X), ...]

    results = []
    used = set()

    for vic in ION_CLASSES:
        virt_z = vic.z
        virt_q = vic.charge

        # A-site charges must match virtual cation charges
        if charge_spec[0][1] != virt_q:
            continue

        # Collect metal candidates matching B-site charges
        metal_combos = []
        for b_spec in charge_spec[1:-1]:  # all B-type sites
            b_count, b_q = b_spec
            candidates = METALS.get(f"+{b_q}", [])
            if not candidates:
                candidates = METALS.get(f"+{abs(b_q)}", [])

        # Generate metal combinations
        b_candidates_all = []
        for b_spec in charge_spec[1:-1]:
            b_count, b_q = b_spec
            candidates = [m for m in METALS.get(f"+{b_q}", []) if m not in {"Fr", "Ra", "Ac", "Pa", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}]
            b_candidates_all.append(candidates)

        # Anion choices
        x_spec = charge_spec[-1]
        x_count, x_q = x_spec
        if abs(x_q) == 1:
            an_candidates = ANIONS["halide"]
        elif abs(x_q) == 2 and template.family == "oxide":
            an_candidates = ANIONS["oxide"]
        elif abs(x_q) == 2 and template.family == "sulfide":
            an_candidates = ANIONS["sulfide"]
        else:
            an_candidates = ANIONS["halide"] + ANIONS["oxide"] + ANIONS["sulfide"]

        # Iterate over metal combos
        def gen_metal_combos(candidates_list):
            if not candidates_list:
                yield []
                return
            first = candidates_list[0]
            rest = candidates_list[1:]
            for combo in gen_metal_combos(rest):
                for m in first[:4]:  # limit per site
                    yield [m] + combo

        count = 0
        for metal_combo in gen_metal_combos(b_candidates_all):
            if count >= n_metal_combos:
                break
            for anion in an_candidates[:3]:
                key = f"{virt_z}_{'_'.join(metal_combo)}_{anion}"
                if key in used:
                    continue
                used.add(key)
                results.append({
                    "template": template.name,
                    "family": template.family,
                    "virtual_z": virt_z,
                    "virtual_charge": virt_q,
                    "metals": metal_combo,
                    "anion": anion,
                    "formula": f"X{virt_z}_{'_'.join(metal_combo)}_{anion}",
                })
                count += 1

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Crystal structure builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_structure_from_template(
    template: Template,
    comp: dict,
) -> Structure:
    """Build a pymatgen Structure from template + composition."""
    a = template.lattice_scale * (1 + 0.1 * np.random.randn())
    b = a * (1 + 0.05 * np.random.randn())
    c = a * (1 + 0.05 * np.random.randn())

    lattice = Lattice.from_parameters(
        a=a, b=b, c=c,
        alpha=template.angles[0], beta=template.angles[1], gamma=template.angles[2],
    )

    # Build species list
    species = []
    site_idx = 0
    metal_idx = 1 if len(comp["metals"]) > 1 else 0

    for label in template.site_labels:
        if label == "A":
            species.append(DummySpecies(get_virtual_symbol(comp["virtual_z"]), oxidation_state=0))
        elif label in ("B", "Bp"):
            m = comp["metals"][min(metal_idx, len(comp["metals"])-1)]
            species.append(Element(m))
            metal_idx += 1
        elif label == "X":
            species.append(Element(comp["anion"]))
        else:
            species.append(Element(comp["anion"]))

    sites = [
        PeriodicSite(species=sp, coords=list(pos), lattice=lattice)
        for sp, pos in zip(species, template.sites_frac)
    ]

    return Structure.from_sites(sites)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dataset cache writer
# ═══════════════════════════════════════════════════════════════════════════════

def save_cache(structures: List[dict], output_dir: Path, split: str):
    """Save structures to CrystalDataset cache format."""
    cache_dir = output_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_pos = np.row_stack([s["pos"] for s in structures])
    all_cell = np.array([s["cell"] for s in structures])
    all_z = np.concatenate([s["atomic_numbers"] for s in structures])
    all_natoms = np.array([s["num_atoms"] for s in structures], dtype=np.int64)
    all_sid = np.arange(len(structures), dtype=np.int64)

    np.save(cache_dir / "pos.npy", all_pos)
    np.save(cache_dir / "cell.npy", all_cell)
    np.save(cache_dir / "atomic_numbers.npy", all_z)
    np.save(cache_dir / "num_atoms.npy", all_natoms)
    np.save(cache_dir / "structure_id.npy", all_sid)

    # Stats
    n_real = int((all_z <= MAX_ATOMIC_NUM).sum())
    n_virt = int((all_z >= VIRTUAL_ELEMENT_OFFSET).sum())
    real_elements = sorted(set(int(z) for z in all_z if z <= MAX_ATOMIC_NUM))
    virt_elements = sorted(set(int(z) for z in all_z if z >= VIRTUAL_ELEMENT_OFFSET))

    metadata = {
        "n_structures": len(structures),
        "n_atoms_total": int(all_pos.shape[0]),
        "split": split,
        "n_real_atoms": n_real,
        "n_virtual_atoms": n_virt,
        "real_elements": real_elements,
        "virtual_elements": virt_elements,
    }
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  {split}: {len(structures):5d} structures, {n_real:5d} real, {n_virt:4d} virtual")
    print(f"    Real Z: {real_elements}")
    print(f"    Virt Z: {virt_elements}")
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_expanded_dataset(
    output_dir: str,
    max_per_template: int = 150,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """Build the expanded multi-family dataset."""
    output_dir = Path(output_dir)
    all_structures = []

    print("=" * 60)
    print("  Expanded Hybrid Crystal Dataset Builder")
    print("=" * 60)

    for template in ALL_TEMPLATES:
        print(f"\n[Template] {template.name} ({template.family})")
        print(f"  Formula: {template.stoichiometry}, Charge: {template.charge_formula}")

        compositions = enumerate_charge_balanced_compositions(
            template, n_metal_combos=max_per_template
        )
        print(f"  Generated {len(compositions)} charge-balanced compositions")

        for comp in tqdm(compositions[:max_per_template], desc=f"  Building {template.name}"):
            try:
                struct = build_structure_from_template(template, comp)

                # Extract numpy data
                frac_coords = struct.frac_coords.astype(np.float32)
                cell = struct.lattice.matrix.astype(np.float32)

                def _get_z(site):
                    s = str(site.specie.symbol)
                    if s.startswith("X") and len(s) > 1 and s[1:].isdigit():
                        return int(s[1:])
                    return site.specie.Z

                atomic_numbers = np.array([_get_z(site) for site in struct], dtype=np.int64)

                all_structures.append({
                    "pos": frac_coords,
                    "cell": cell,
                    "atomic_numbers": atomic_numbers,
                    "num_atoms": len(struct),
                })
            except Exception as e:
                pass  # skip problematic combos

    total = len(all_structures)
    print(f"\n{'=' * 60}")
    print(f"  Total structures generated: {total}")
    print(f"{'=' * 60}")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(all_structures)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    for split_name, split_data in [
        ("train", all_structures[:n_train]),
        ("val", all_structures[n_train:n_train + n_val]),
        ("test", all_structures[n_train + n_val:]),
    ]:
        save_cache(split_data, output_dir, split_name)

    print(f"\nDone! Dataset saved to {output_dir}/")
    print(f"  train: {n_train}, val: {n_val}, test: {total - n_train - n_val}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./datasets/hybrid_expanded")
    parser.add_argument("--max-per-template", type=int, default=150)
    parser.add_argument("--build-all-splits", action="store_true")
    args = parser.parse_args()

    build_expanded_dataset(
        output_dir=args.output_dir,
        max_per_template=args.max_per_template,
    )
