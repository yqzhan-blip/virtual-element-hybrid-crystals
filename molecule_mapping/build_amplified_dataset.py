#!/usr/bin/env python3
"""
Amplified dataset builder — generates ~5500+ new structures to reach 8000 total.
Adds many more metals, templates, and anion combinations.

Usage: python build_amplified_dataset.py --output-dir ../datasets/hybrid_amplified
"""

import hashlib, json, sys, os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from pymatgen.core import Lattice, PeriodicSite, Element, DummySpecies, Structure

# Path setup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM, VIRTUAL_ELEMENT_OFFSET, get_virtual_symbol,
)
from mattergen.molecule_mapping.ion_classes import ION_CLASSES

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Expanded element database (~45 metals across all oxidation states)
# ═══════════════════════════════════════════════════════════════════════════════

METALS_BY_CHARGE = {
    +1: ["Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Tl", "In", "Hg"],
    +2: ["Be", "Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Pb", "Sn", "Ge", "Cr", "V", "Pd", "Pt", "Hg", "Eu", "Sm"],
    +3: ["Al", "Ga", "In", "Sc", "Y", "La", "Cr", "Mn", "Fe", "Co", "Ni", "Bi", "Sb", "Ru", "Rh", "Ir", "Au", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
    +4: ["Ti", "Zr", "Hf", "Sn", "Ge", "Mn", "Ce", "Th", "U", "V", "Nb", "Ta", "Mo", "W", "Ru", "Os", "Ir", "Pt", "Pb", "Si"],
    +5: ["Nb", "Ta", "Sb", "Bi", "V", "Mo", "W", "Re", "U"],
    +6: ["Cr", "Mo", "W", "U", "Re", "Os"],
}

# Cations directly usable from virtual element classes (all +1 and +2 are most common)
ANIONS_BY_FAMILY = {
    "halide": ["F", "Cl", "Br", "I"],
    "oxide": ["O"],
    "sulfide": ["S", "Se", "Te"],
    "mixed-OF": ["O", "F"],    # oxyfluorides
    "mixed-SO": ["S", "O"],    # oxysulfides
    "mixed-SSe": ["S", "Se"],  # sulfoselenides
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Expanded template library (20+ templates)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Template:
    name: str
    family: str
    stoichiometry: dict  # {A: count_A, B: count_B, X: count_X}
    charge_spec: list   # [(count, charge), ...] for each site type
    lattice_scale: float
    sites_frac: list
    site_labels: list
    angles: tuple = (90, 90, 90)

    @property
    def n_atoms(self) -> int:
        return sum(self.stoichiometry.values())


def make_sites(template):
    """Build site list with correct multiplicity."""
    species_labels = template.site_labels
    return template.sites_frac, species_labels


# ──────────────────────────────────────────────────────────────────────────────
# All templates combined
TEMPLATES = []

# === Halides (6 templates) ===
T = Template("ABX3_perov", "halide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-1)], 6.0,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","X","X","X"])
TEMPLATES.append(T)

T = Template("A2BX4_RP", "halide", {"A":2,"B":1,"X":4}, [(2,1),(1,2),(4,-1)], 6.0,
             [(0,0,0),(0,0,0.5),(0.5,0.5,0.25),(0.5,0,0.25),(0,0.5,0.25),(0.5,0.5,0.75),(0.5,0,0.75),(0,0.5,0.75)],
             ["A","A","B","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

T = Template("A3B2X9_2D", "halide", {"A":3,"B":2,"X":9}, [(3,1),(2,3),(9,-1)], 8.0,
             [(0,0,0),(0.5,0.5,0),(0,0,0.5),(0.33,0.33,0.25),(0.67,0.67,0.75),
              (0.33,0,0.25),(0.67,0,0.75),(0,0.33,0.25),(0,0.67,0.75),
              (0.33,0.33,0),(0.67,0.67,0.5),(0.33,0,0),(0.67,0,0.5),(0,0.33,0),(0,0.67,0.5)],
             ["A","A","A","B","B"]+["X"]*9)
TEMPLATES.append(T)

T = Template("A2BX6_elpas", "halide", {"A":2,"B":1,"X":6}, [(2,1),(1,4),(6,-1)], 10.5,
             [(0.25,0.25,0.25),(0.75,0.75,0.75),(0,0,0),
              (0.25,0,0),(0.75,0,0),(0,0.25,0),(0,0.75,0),(0,0,0.25),(0,0,0.75)],
             ["A","A","B","X","X","X","X","X","X"])
TEMPLATES.append(T)

T = Template("A4BX6_2D", "halide", {"A":4,"B":1,"X":6}, [(4,1),(1,2),(6,-1)], 8.5,
             [(0,0,0),(0,0,0.33),(0,0,0.67),(0.5,0.5,0.17),
              (0.5,0.5,0.5),
              (0.25,0.25,0.08),(0.25,0.75,0.08),(0.75,0.25,0.08),(0.75,0.75,0.08),
              (0.25,0.25,0.42),(0.25,0.75,0.42),(0.75,0.25,0.42),(0.75,0.75,0.42)],
             ["A","A","A","A","B","X","X","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

# AA'BX₄ — mixed A-site 2D perovskite
T = Template("AApBX4_mix", "halide", {"A":1,"Ap":1,"B":1,"X":4}, [(1,1),(1,1),(1,2),(4,-1)], 6.5,
             [(0,0,0),(0.5,0.5,0.5),(0,0,0.25),(0.5,0,0.25),(0,0.5,0.25),
              (0.5,0.5,0.75),(0.5,0,0.75),(0,0.5,0.75)],
             ["A","Ap","B","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

# === Oxides (8 templates) ===
T = Template("ABO3_perov", "oxide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-2)], 3.9,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","O","O","O"])
TEMPLATES.append(T)

T = Template("AB2O4_spinel", "oxide", {"A":1,"B":2,"X":4}, [(1,2),(2,3),(4,-2)], 8.3,
             [(0.125,0.125,0.125),(0.5,0.5,0.5),(0.5,0.75,0.75),
              (0.25,0.25,0.25),(0.25,0.25,0.75),(0.25,0.75,0.25),(0.25,0.75,0.75)],
             ["A","B","B","O","O","O","O"])
TEMPLATES.append(T)

T = Template("A2B2O7_pyro", "oxide", {"A":2,"B":2,"X":7}, [(2,3),(2,4),(7,-2)], 10.2,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0.5),(0.25,0.75,0.75),
              (0.125,0.125,0.125),(0.125,0.625,0.625),(0.625,0.125,0.625),(0.625,0.625,0.125),
              (0.375,0.375,0.375),(0.375,0.875,0.875),(0.875,0.375,0.875)],
             ["A","A","B","B","O","O","O","O","O","O","O"])
TEMPLATES.append(T)

T = Template("A2BBpO6_DP", "oxide", {"A":2,"B":1,"Bp":1,"X":6}, [(2,1),(1,2),(1,5),(6,-2)], 7.9,
             [(0,0,0),(0.5,0.5,0.5),(0,0,0),(0.5,0.5,0.5),
              (0.25,0,0),(0.75,0,0),(0,0.25,0),(0,0.75,0),(0,0,0.25),(0,0,0.75)],
             ["A","A","B","Bp","O","O","O","O","O","O"])
TEMPLATES.append(T)

T = Template("ABO3_hex", "oxide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-2)], 5.6,
             [(0,0,0),(0.33,0.67,0.5),(0.17,0.83,0.25),(0.83,0.17,0.25),
              (0.17,0.83,0.75),(0.83,0.17,0.75)],
             ["A","B","O","O","O","O"], (90,90,120))
TEMPLATES.append(T)

T = Template("ABO2F_perov", "mixed-OF", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(2,-2),(1,-1)], 3.9,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","O","O","F"])
TEMPLATES.append(T)

T = Template("A3B2O8_garnet", "oxide", {"A":3,"B":2,"X":8}, [(3,2),(2,3),(8,-2)], 12.0,
             [(0,0,0),(0.25,0.5,0),(0.5,0.25,0),(0.375,0,0.75),(0.625,0,0.75),
              (0,0.125,0),(0,0.375,0),(0,0.625,0),(0,0.875,0),
              (0.25,0,0),(0.75,0,0),(0.5,0.25,0.5),(0.5,0.75,0.5)],
             ["A","A","A","B","B","O","O","O","O","O","O","O","O"])
TEMPLATES.append(T)

T = Template("A2BO4_K2NiF4", "oxide", {"A":2,"B":1,"X":4}, [(2,1),(1,2),(4,-2)], 3.9,
             [(0,0,0),(0.5,0.5,0.5),(0,0,0.35),(0,0,0.65),
              (0.5,0,0.15),(0,0.5,0.15),(0.5,0,0.85),(0,0.5,0.85)],
             ["A","A","B","O","O","O","O","O"], (90,90,90))
TEMPLATES.append(T)

# === Sulfides (6 templates) ===
T = Template("ABS3_perov", "sulfide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-2)], 5.6,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","S","S","S"])
TEMPLATES.append(T)

T = Template("AB2S4_thio", "sulfide", {"A":1,"B":2,"X":4}, [(1,2),(2,3),(4,-2)], 10.0,
             [(0.125,0.125,0.125),(0.5,0.5,0.5),(0.5,0.75,0.75),
              (0.25,0.25,0.25),(0.25,0.25,0.75),(0.25,0.75,0.25),(0.25,0.75,0.75)],
             ["A","B","B","S","S","S","S"])
TEMPLATES.append(T)

T = Template("AB2Se4_thio", "sulfide", {"A":1,"B":2,"X":4}, [(1,2),(2,3),(4,-2)], 10.2,
             [(0.125,0.125,0.125),(0.5,0.5,0.5),(0.5,0.75,0.75),
              (0.25,0.25,0.25),(0.25,0.25,0.75),(0.25,0.75,0.25),(0.25,0.75,0.75)],
             ["A","B","B","Se","Se","Se","Se"])
TEMPLATES.append(T)

T = Template("ABTe3_perov", "sulfide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-2)], 6.2,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","Te","Te","Te"])
TEMPLATES.append(T)

T = Template("A2B2S7_pyro", "sulfide", {"A":2,"B":2,"X":7}, [(2,2),(2,4),(7,-2)], 10.5,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0.5),(0.25,0.75,0.75),
              (0.125,0.125,0.125),(0.125,0.625,0.625),(0.625,0.125,0.625),(0.625,0.625,0.125),
              (0.375,0.375,0.375),(0.375,0.875,0.875),(0.875,0.375,0.875)],
             ["A","A","B","B","S","S","S","S","S","S","S"])
TEMPLATES.append(T)

# Mixed S/Se sulfide template
T = Template("ABSSe2_mix", "sulfide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(1,-2),(2,-2)], 5.8,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","S","Se","Se"])
TEMPLATES.append(T)
# Additional +1-friendly templates for maximizing count (matching most virtual classes)

# Halide: Aurivillius-like layered
T = Template("A2B2X7_auri", "halide", {"A":2,"B":2,"X":7}, [(2,1),(2,3),(7,-1)], 8.5,
             [(0,0,0),(0.5,0.5,0.5),(0,0,0.25),(0.5,0.5,0.75),
              (0.25,0,0.12),(0,0.25,0.12),(0.75,0,0.12),(0,0.75,0.12),
              (0.25,0,0.37),(0,0.25,0.37),(0.75,0,0.37),(0,0.75,0.37)],
             ["A","A","B","B"]+["X"]*7, (90,90,90))
TEMPLATES.append(T)

# Halide: A5B2X11 layered
T = Template("A5B2X11_lay", "halide", {"A":5,"B":2,"X":11}, [(5,1),(2,2),(11,-1)], 9.0,
             [(0,0,0),(0,0,0.25),(0,0,0.5),(0.5,0.5,0.12),(0.5,0.5,0.62),
              (0,0,0.12),(0,0,0.62),
              (0.25,0.25,0.06),(0.25,0.75,0.06),(0.75,0.25,0.06),(0.75,0.75,0.06),
              (0.25,0.25,0.31),(0.25,0.75,0.31),(0.75,0.25,0.31),(0.75,0.75,0.31),
              (0.25,0.25,0.56),(0.25,0.75,0.56),(0.75,0.25,0.56),(0.75,0.75,0.56)],
             ["A","A","A","A","A","B","B","X","X","X","X","X","X","X","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

# Oxide: A3B4O12 perovskite-related
T = Template("A3B4O12", "oxide", {"A":3,"B":4,"X":12}, [(3,1),(4,2),(12,-2)], 7.5,
             [(0,0,0),(0.5,0.5,0),(0,0,0.5),
              (0.25,0.25,0.25),(0.75,0.25,0.25),(0.25,0.75,0.25),(0.75,0.75,0.25),
              (0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0.5,0.5),
              (0.5,0,0.25),(0,0.5,0.25),(0,0,0.25),(0.5,0.5,0.25),
              (0.5,0,0.75),(0,0.5,0.75),(0,0,0.75),(0.5,0.5,0.75)],
             ["A","A","A","B","B","B","B","O","O","O","O","O","O","O","O","O","O","O","O"], (90,90,90))
TEMPLATES.append(T)

# Oxide: ABO3 orthorhombic
T = Template("ABO3_ortho", "oxide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-2)], 5.5,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0,0),(0,0.5,0),(0,0,0.5)],
             ["A","B","O","O","O"], (90,90,90))
TEMPLATES.append(T)

# Sulfide: A2BS4
T = Template("A2BS4", "sulfide", {"A":2,"B":1,"X":4}, [(2,1),(1,2),(4,-2)], 6.5,
             [(0,0,0),(0.5,0.5,0.5),(0.25,0.25,0.25),
              (0.25,0.75,0.25),(0.75,0.25,0.25),(0.75,0.75,0.25),(0.25,0.25,0.75)],
             ["A","A","B","S","S","S","S"])
TEMPLATES.append(T)

# Sulfide: A3BS4
T = Template("A3BS4", "sulfide", {"A":3,"B":1,"X":4}, [(3,1),(1,3),(4,-2)], 7.0,
             [(0,0,0),(0.5,0.5,0),(0,0,0.5),(0.5,0.5,0.5),
              (0.25,0.25,0.25),(0.25,0.75,0.25),(0.75,0.25,0.25),(0.75,0.75,0.25)],
             ["A","A","A","B","S","S","S","S"])
TEMPLATES.append(T)

# Halide: A4B2X10 chain
T = Template("A4B2X10", "halide", {"A":4,"B":2,"X":10}, [(4,1),(2,2),(10,-1)], 7.0,
             [(0,0,0),(0,0,0.5),(0.5,0.5,0.25),(0.5,0.5,0.75),
              (0.25,0.25,0.25),(0.25,0.75,0.75),
              (0.1,0.1,0.1),(0.1,0.6,0.1),(0.6,0.1,0.1),(0.6,0.6,0.1),
              (0.1,0.1,0.4),(0.1,0.6,0.4),(0.6,0.1,0.4),(0.6,0.6,0.4),
              (0.1,0.1,0.6),(0.1,0.6,0.6),(0.6,0.1,0.6),(0.6,0.6,0.6)],
             ["A","A","A","A","B","B","X","X","X","X","X","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

# Halide: ABX3_tetr (tetragonal perovskite)
T = Template("ABX3_tetr", "halide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(3,-1)], 8.9,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0,0.25),(0,0.5,0.25),(0.5,0.5,0)],
             ["A","B","X","X","X"], (90,90,90))
TEMPLATES.append(T)

# Mixed: Oxyhalide perovskite
T = Template("ABO2X_oxyhal", "halide", {"A":1,"B":1,"X":3}, [(1,1),(1,2),(2,-2),(1,-1)], 5.0,
             [(0,0,0),(0.5,0.5,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
             ["A","B","O","O","F"])
TEMPLATES.append(T)

# Halide: Cs3Bi2I9-type vacancy ordered
T = Template("A3B2X9_vac", "halide", {"A":3,"B":2,"X":9}, [(3,1),(2,3),(9,-1)], 9.0,
             [(0,0,0),(0.33,0.33,0.33),(0.67,0.67,0.67),
              (0.17,0.17,0.17),(0.83,0.83,0.83),
              (0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5),
              (0.17,0.5,0.5),(0.83,0.5,0.5),(0.5,0.17,0.17),(0.5,0.83,0.83)],
             ["A","A","A","B","B"]+["X"]*9, (90,90,90))
TEMPLATES.append(T)

# Halide: A2B2X6 ditelluride-type
T = Template("A2B2X6_di", "halide", {"A":2,"B":2,"X":6}, [(2,1),(2,2),(6,-1)], 7.5,
             [(0,0,0),(0.5,0.5,0.5),(0.25,0.25,0.25),(0.75,0.75,0.75),
              (0.5,0,0.25),(0,0.5,0.25),(0,0,0.25),(0.5,0.5,0.25),
              (0.5,0,0.75),(0,0.5,0.75)],
             ["A","A","B","B","X","X","X","X","X","X"], (90,90,90))
TEMPLATES.append(T)

print(f"[INFO] Total templates: {len(TEMPLATES)}")
for t in TEMPLATES:
    print(f"  {t.name:20s}  {t.family:10s}  {t.stoichiometry}  charges={[f'{n}@{q:+d}' for n,q in t.charge_spec]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Composition enumerator — relaxed charge matching
# ═══════════════════════════════════════════════════════════════════════════════

def enumerate_all_compositions(template, max_per_virtual=200):
    """Generate ALL reasonable compositions for a template, matching virtual Z charges
    to metal charges via stoichiometry. Uses all virtual classes, all viable metals."""
    results = []

    for vic in ION_CLASSES:
        v_z = vic.z
        v_q = vic.charge

        # Match virtual charge to A-site charge in template (with ±1 tolerance)
        a_spec = template.charge_spec[0]
        a_charge = a_spec[1]
        # Relaxed matching: allow exact match or one charge difference
        if abs(a_charge - v_q) > 1:
            continue  # too far apart — skip this virtual class for this template

        # Get metal candidates for each B-site
        b_specs = [s for s in template.charge_spec[1:-1]]  # all metal sites
        x_spec = template.charge_spec[-1]  # anion

        # For each B-site, collect all viable metals (expanded to 15)
        metal_options = []
        for count, target_q in b_specs:
            candidates = [m for m in METALS_BY_CHARGE.get(target_q, []) if m not in {"Fr","Ra","Ac","Pa","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"} and len(m) <= 2]
            if not candidates:
                # Fallback: use any metal with this charge or nearby
                for fallback_q in [target_q + d for d in [0, 1, -1, 2, -2]]:
                    candidates = [m for m in METALS_BY_CHARGE.get(fallback_q, []) if m not in {"Fr","Ra","Ac","Pa","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"} and len(m) <= 2]
                    if candidates:
                        break
            metal_options.append(candidates[:15])  # expanded from 8 to 15

# Mixed templates: ABO2F_perov (A,B,O,O,F) — O and F are separate anion types
# For mixed anion templates, we need to handle the X-site differently
# The charge_spec for mixed templates has multiple X entries, but we treat them as a single "anion group"
# So we sum the anion charges and match with a single anion type

        # Anion selection (expanded to 4 for all)
        x_q = x_spec[1]
        if abs(x_q) == 1:
            anion_candidates = ["F","Cl","Br","I"]
        elif abs(x_q) == 2:
            if "oxide" in template.family:
                anion_candidates = ["O","S","Se"]  # oxides + some sulfide overlap
            elif "sulfide" in template.family:
                anion_candidates = ["S","Se","Te","O"]
            else:
                anion_candidates = ["O","S","Se","Te"]
        else:
            # Mixed anion templates (e.g., ABO2F has O@+2 and F@-1)
            # We treat the mixed anion as a single "effective anion" for structure building
            # The actual anion assignment is handled in build_structure by matching labels
            anion_candidates = ["O","F","S","Cl"]

        # Generate combinations
        seen = set()
        np.random.seed(42)

        # For single B-site: enumerate all metals × all anions
        if len(metal_options) == 1:
            for metal in metal_options[0]:
                for anion in anion_candidates:
                    key = f"{v_z}_{metal}_{anion}"
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "template": template.name, "family": template.family,
                        "virtual_z": v_z, "virtual_charge": v_q,
                        "metals": [metal], "anion": anion,
                    })
                    if len(results) >= max_per_virtual * len(ION_CLASSES) * 3:
                        break
                if len(results) >= max_per_virtual * len(ION_CLASSES) * 3:
                    break

        # For two B-sites: enumerate combinations
        elif len(metal_options) == 2:
            m1, m2 = metal_options
            combos = []
            for a in m1[:10]:  # expanded from 5 to 10
                for b in m2[:10]:
                    combos.append((a, b))
            np.random.shuffle(combos)
            for m1_sel, m2_sel in combos[:50]:  # expanded from 20 to 50
                for anion in anion_candidates:
                    key = f"{v_z}_{m1_sel}_{m2_sel}_{anion}"
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "template": template.name, "family": template.family,
                        "virtual_z": v_z, "virtual_charge": v_q,
                        "metals": [m1_sel, m2_sel], "anion": anion,
                    })

        # For three or more B-sites: pick representative metals
        else:
            for i, mo in enumerate(metal_options[:3]):
                m = mo[0] if mo else "Ca"
                combos = [[m]]
            # Just pick one metal per site
            combo = [mo[0] for mo in metal_options[:3]]
            for anion in anion_candidates[:4]:  # expanded from 2 to 4
                key = f"{v_z}_{'_'.join(combo)}_{anion}"
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "template": template.name, "family": template.family,
                    "virtual_z": v_z, "virtual_charge": v_q,
                    "metals": combo, "anion": anion,
                })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Structure builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_structure(template, comp):
    """Build structure from template + composition with lattice perturbation."""
    a_scale = template.lattice_scale * (1 + 0.15 * np.random.randn())
    b_scale = template.lattice_scale * (1 + 0.10 * np.random.randn())
    c_scale = template.lattice_scale * (1 + 0.10 * np.random.randn())

    lattice = Lattice.from_parameters(
        a=a_scale, b=b_scale, c=c_scale,
        alpha=template.angles[0], beta=template.angles[1], gamma=template.angles[2],
    )

    # Assign species by matching labels to composition
    species_list = []
    metal_ptr = 0
    for label in template.site_labels:
        if label == "A" or label == "Ap":
            species_list.append(DummySpecies(get_virtual_symbol(comp["virtual_z"]), oxidation_state=0))
        elif label == "B" or label == "Bp":
            m = comp["metals"][min(metal_ptr, len(comp["metals"]) - 1)]
            species_list.append(Element(m))
            metal_ptr += 1
        elif label in ("O", "F", "S", "Se", "Te"):
            species_list.append(Element(label))
        else:
            species_list.append(Element(comp["anion"]))

    sites = [PeriodicSite(sp, list(pos), lattice) for sp, pos in zip(species_list, template.sites_frac)]
    return Structure.from_sites(sites)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Save cache
# ═══════════════════════════════════════════════════════════════════════════════

def save_cache(structures, output_dir, split):
    cache_dir = output_dir / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_pos = np.vstack([s["pos"] for s in structures])
    all_cell = np.array([s["cell"] for s in structures])
    all_z = np.concatenate([s["atomic_numbers"] for s in structures])
    all_natoms = np.array([s["num_atoms"] for s in structures], dtype=np.int64)
    all_sid = np.arange(len(structures), dtype=np.int64)

    np.save(cache_dir / "pos.npy", all_pos)
    np.save(cache_dir / "cell.npy", all_cell)
    np.save(cache_dir / "atomic_numbers.npy", all_z)
    np.save(cache_dir / "num_atoms.npy", all_natoms)
    np.save(cache_dir / "structure_id.npy", all_sid)

    n_real = int((all_z <= MAX_ATOMIC_NUM).sum())
    n_virt = int((all_z >= VIRTUAL_ELEMENT_OFFSET).sum())
    real_z = sorted(set(int(z) for z in all_z if z <= MAX_ATOMIC_NUM))
    virt_z = sorted(set(int(z) for z in all_z if z >= VIRTUAL_ELEMENT_OFFSET))

    meta = {"n_structures": len(structures), "split": split,
            "n_real_atoms": n_real, "n_virtual_atoms": n_virt,
            "real_elements": real_z, "virtual_elements": virt_z}
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {split}: {len(structures):5d} structs, {n_real:5d} real ({len(real_z)} elem), {n_virt:4d} virtual ({len(virt_z)} classes)")
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════════

def build_amplified(output_dir, target_total=8000):
    output_dir = Path(output_dir)
    all_structs = []

    print(f"Building amplified dataset (target: ~{target_total - 2456} new structures)")
    print(f"Templates: {len(TEMPLATES)}, Virtual classes: {len(ION_CLASSES)}")

    for template in TEMPLATES:
        comps = enumerate_all_compositions(template, max_per_virtual=30)
        n_build = min(len(comps), 800)

        for comp in comps[:n_build]:
            try:
                struct = build_structure(template, comp)
                fc = struct.frac_coords.astype(np.float32)
                cell = struct.lattice.matrix.astype(np.float32)

                def _z(site):
                    s = str(site.specie.symbol)
                    return int(s[1:]) if (s.startswith("X") and len(s) > 1 and s[1:].isdigit()) else site.specie.Z

                z_arr = np.array([_z(s) for s in struct], dtype=np.int64)
                all_structs.append({"pos": fc, "cell": cell, "atomic_numbers": z_arr, "num_atoms": len(struct)})
            except:
                pass

        print(f"  {template.name:20s}: {n_build:4d} compositions → {len(all_structs):5d} cumulative")

    total = len(all_structs)
    np.random.seed(42)
    np.random.shuffle(all_structs)

    train_n = int(total * 0.8)
    val_n = int(total * 0.1)

    print(f"\nTotal: {total} new structures")
    for name, data in [("train", all_structs[:train_n]), ("val", all_structs[train_n:train_n+val_n]),
                        ("test", all_structs[train_n+val_n:])]:
        save_cache(data, output_dir, name)
    print(f"\nDone! → {output_dir}/")


if __name__ == "__main__":
    build_amplified("../datasets/hybrid_amplified")
