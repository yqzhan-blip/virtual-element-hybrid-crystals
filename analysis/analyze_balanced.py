#!/usr/bin/env python3
"""
筛选并分析97个电荷平衡结构（高质量候选）。

1. 从 validation stats.json 中筛选 balanced=True 的结构
2. 分析元素组合模式（金属-卤素-有机阳离子）
3. 识别典型钙钛矿框架
4. 统计模板分布
5. 输出筛选后的CIF到 paper_results/balanced/
6. 生成统计报告和图表
"""

import os, sys, json, shutil
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from pymatgen.core import Composition, Element
from pymatgen.io.cif import CifParser

# ── Config ─────────────────────────────────────────────────────────────
DECODED_DIR = Path("C:/Users/zhan/WorkBuddy/2026-05-28-task-12/mattergen/results_hybrid_8k_1000/decoded_v2")
VALIDATION_STATS = Path("C:/Users/zhan/WorkBuddy/2026-05-28-task-12/mattergen/results_hybrid_8k_1000/validation/stats.json")
OUTPUT_DIR = Path("C:/Users/zhan/WorkBuddy/2026-05-28-task-12/mattergen/results_hybrid_8k_1000/balanced")
REPORT_PATH = OUTPUT_DIR / "balanced_analysis.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load validation stats ────────────────────────────────────────────
with open(VALIDATION_STATS, "r") as f:
    stats = json.load(f)

# ── Identify balanced structures ──────────────────────────────────────
balanced_examples = [ex for ex in stats["formula_examples"] if ex["balanced"]]

# We only have 20 examples in stats. Need to scan all 1000 for complete list.
# Let's read the log file to get all balanced structures, or re-scan.

# ── Re-scan all decoded CIFs for balanced structures ──────────────────
OXIDATION_STATES = {
    "H": +1, "Li": +1, "Na": +1, "K": +1, "Rb": +1, "Cs": +1,
    "Be": +2, "Mg": +2, "Ca": +2, "Sr": +2, "Ba": +2, "Zn": +2,
    "Al": +3, "Ga": +3, "In": +3, "La": +3, "Ce": +3, "Bi": +3, "Sb": +3,
    "Ti": +4, "Zr": +4, "Hf": +4, "Th": +4, "U": +4, "Sn": +4, "Pb": +2,
    "V": +3, "Cr": +3, "Mn": +2, "Fe": +2, "Co": +2, "Ni": +2, "Cu": +2,
    "Y": +3, "Nb": +5, "Mo": +6, "Ru": +3, "Rh": +3, "Ag": +1, "Cd": +2,
    "W": +6, "Re": +7, "Hg": +2, "Tl": +1, "Ta": +5, "Sc": +3,
    "O": -2, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1,
}

METALS = {"Pb", "Sn", "Ge", "Ti", "Bi", "Sb", "Al", "Ga", "In", "La", "Ce", "Y", "U", "Th", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg", "Ag", "Tl", "Ru", "Rh", "Re", "Sc", "Ba", "Sr", "Ca", "Mg", "Be", "Li", "Na", "K", "Rb", "Cs"}
HALIDES = {"F", "Cl", "Br", "I"}

def validate_charge(comp_dict):
    """Calculate net charge for a composition dict."""
    organic_charge = 0
    inorganic_charge = 0
    has_organic = False
    
    for el, amt in comp_dict.items():
        if el in ["C", "H"]:
            has_organic = True
        elif el == "N":
            has_organic = True
            organic_charge += amt * 1  # +1 per N (protonated)
        else:
            oxi = OXIDATION_STATES.get(el, 0)
            inorganic_charge += oxi * amt
    
    net = organic_charge + inorganic_charge
    return net, has_organic, organic_charge, inorganic_charge

def analyze_structure(cif_path):
    """Analyze a single decoded structure."""
    try:
        parser = CifParser(str(cif_path))
        structs = parser.get_structures()
        if not structs:
            return None
        struct = structs[0]
        comp = struct.composition
        comp_dict = {str(el): amt for el, amt in comp.items()}
        
        net, has_organic, org_charge, inorg_charge = validate_charge(comp_dict)
        balanced = abs(net) < 0.5
        
        # Identify metals and halides
        metals = [el for el in comp_dict if el in METALS]
        halides = [el for el in comp_dict if el in HALIDES]
        
        # Identify template type (from formula N count and H count)
        n_count = comp_dict.get("N", 0)
        h_count = comp_dict.get("H", 0)
        c_count = comp_dict.get("C", 0)
        
        # Estimate organic cation class
        if n_count == 0 and c_count == 0:
            template_type = "Inorganic (no organic)"
        elif n_count == 1 and c_count <= 2:
            template_type = "A3D-small (MA/FA)"
        elif n_count == 1 and c_count > 2 and c_count <= 5:
            template_type = "Sp2D-alkyl (BA/HA)"
        elif n_count == 1 and c_count > 5:
            template_type = "Sp2D-aromatic (PEA)"
        elif n_count == 2 and h_count <= 12:
            template_type = "DiA-alkyl (EDA)"
        elif n_count == 2 and h_count > 12:
            template_type = "DiA-aromatic (XDA)"
        elif n_count == 3:
            template_type = "TriA (TAPA)"
        elif n_count == 4:
            template_type = "TetraA (TETA)"
        elif n_count >= 5:
            template_type = "PentaHexaA (PEHA)"
        else:
            template_type = "Unknown"
        
        return {
            "name": cif_path.stem,
            "formula": str(struct.formula),
            "composition": comp_dict,
            "net_charge": net,
            "balanced": balanced,
            "organic_charge": org_charge,
            "inorganic_charge": inorg_charge,
            "has_organic": has_organic,
            "metals": metals,
            "halides": halides,
            "template_type": template_type,
            "n_atoms": len(struct),
            "volume": float(struct.volume),
        }
    except Exception as e:
        print(f"  [WARN] Failed to analyze {cif_path}: {e}")
        return None

# ── Main analysis ──────────────────────────────────────────────────────
print("=" * 60)
print("Analyzing all 1000 decoded structures for charge balance...")
print("=" * 60)

cif_files = sorted(DECODED_DIR.glob("*_decoded.cif"))
total = len(cif_files)

all_results = []
balanced_results = []

for i, cif_path in enumerate(cif_files):
    if i % 100 == 0:
        print(f"  {i}/{total}...")
    result = analyze_structure(cif_path)
    if result:
        all_results.append(result)
        if result["balanced"]:
            balanced_results.append(result)

print(f"\nTotal analyzed: {len(all_results)}")
print(f"Balanced: {len(balanced_results)}")

# ── Copy balanced CIFs ───────────────────────────────────────────────
BALANCED_CIF_DIR = OUTPUT_DIR / "cif"
BALANCED_CIF_DIR.mkdir(exist_ok=True)

for result in balanced_results:
    src = DECODED_DIR / f"{result['name']}.cif"
    dst = BALANCED_CIF_DIR / f"{result['name']}.cif"
    if src.exists():
        shutil.copy(src, dst)

print(f"Copied {len(balanced_results)} balanced CIFs to {BALANCED_CIF_DIR}")

# ── Statistics ───────────────────────────────────────────────────────
def get_metal_halide_pair(result):
    metals = result["metals"]
    halides = result["halides"]
    metal = metals[0] if metals else "None"
    halide = halides[0] if halides else "None"
    return f"{metal}-{halide}"

# Element combination statistics
metal_counter = Counter()
halide_counter = Counter()
pair_counter = Counter()
template_counter = Counter()
formula_counter = Counter()

for result in balanced_results:
    for m in result["metals"]:
        metal_counter[m] += 1
    for h in result["halides"]:
        halide_counter[h] += 1
    pair_counter[get_metal_halide_pair(result)] += 1
    template_counter[result["template_type"]] += 1
    formula_counter[result["formula"]] += 1

# Identify perovskite-like structures (ABX3 or variants)
def is_perovskite_like(result):
    """Check if structure matches ABX3 or common variant formulas."""
    metals = result["metals"]
    halides = result["halides"]
    comp = result["composition"]
    
    # Common perovskite: metal + halide ratio ~ 1:3
    if not metals or not halides:
        return False
    
    metal_count = sum(comp[m] for m in metals)
    halide_count = sum(comp[h] for h in halides)
    
    if metal_count > 0 and halide_count > 0:
        ratio = halide_count / metal_count
        if 2.5 <= ratio <= 3.5:  # Allow some variation from ideal 3
            return True
    return False

perovskite_count = sum(1 for r in balanced_results if is_perovskite_like(r))

# ── Build report ────────────────────────────────────────────────────
report = {
    "metadata": {
        "total_analyzed": len(all_results),
        "balanced_count": len(balanced_results),
        "balanced_percentage": round(len(balanced_results) / len(all_results) * 100, 2),
        "perovskite_like_count": perovskite_count,
        "perovskite_like_percentage": round(perovskite_count / len(balanced_results) * 100, 2) if balanced_results else 0,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    },
    "element_statistics": {
        "metal_distribution": dict(metal_counter),
        "halide_distribution": dict(halide_counter),
        "metal_halide_pairs": dict(pair_counter),
    },
    "template_statistics": dict(template_counter),
    "top_formulas": dict(formula_counter.most_common(20)),
    "balanced_structures": balanced_results,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\nReport saved: {REPORT_PATH}")

# ── Print summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BALANCED STRUCTURES ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total analyzed: {report['metadata']['total_analyzed']}")
print(f"Balanced: {report['metadata']['balanced_count']} ({report['metadata']['balanced_percentage']}%)")
print(f"Perovskite-like: {report['metadata']['perovskite_like_count']} ({report['metadata']['perovskite_like_percentage']}%)")
print(f"\nMetal distribution:")
for metal, count in metal_counter.most_common(10):
    print(f"  {metal}: {count}")
print(f"\nHalide distribution:")
for halide, count in halide_counter.most_common(10):
    print(f"  {halide}: {count}")
print(f"\nTop metal-halide pairs:")
for pair, count in pair_counter.most_common(10):
    print(f"  {pair}: {count}")
print(f"\nTemplate distribution:")
for tmpl, count in template_counter.most_common(10):
    print(f"  {tmpl}: {count}")
print(f"\nTop formulas:")
for formula, count in formula_counter.most_common(10):
    print(f"  {formula}: {count}")
print(f"\nOutput: {OUTPUT_DIR}")
print("=" * 60)
