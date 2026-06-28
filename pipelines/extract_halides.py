import json
import statistics
from collections import Counter

# Read the JSON file
with open("c:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/analysis/balanced_116/balanced_116_analysis.json", "r") as f:
    data = json.load(f)

# Filter structures with halides
halide_structures = [s for s in data if len(s.get("halides", [])) > 0]
total_halide = len(halide_structures)

# Breakdown by halide type (structures can have multiple halides)
halide_counts = Counter()
for s in halide_structures:
    for h in s["halides"]:
        halide_counts[h] += 1

# Breakdown by metal type (individual metals, not combinations)
metal_counts = Counter()
for s in halide_structures:
    for m in s.get("metals", []):
        metal_counts[m] += 1

# Breakdown by template type
template_counts = Counter(s["template_type"] for s in halide_structures)

# Band gap statistics
band_gaps = [s["band_gap_estimate_eV"] for s in halide_structures]
bg_stats = {
    "min": min(band_gaps),
    "max": max(band_gaps),
    "mean": statistics.mean(band_gaps),
    "median": statistics.median(band_gaps),
    "std": statistics.stdev(band_gaps) if len(band_gaps) > 1 else 0.0
}

# CHGNet energy statistics
chgnet_energies = [s["chgnet_energy_per_atom_eV"] for s in halide_structures]
chg_stats = {
    "min": min(chgnet_energies),
    "max": max(chgnet_energies),
    "mean": statistics.mean(chgnet_energies),
    "median": statistics.median(chgnet_energies),
    "std": statistics.stdev(chgnet_energies) if len(chgnet_energies) > 1 else 0.0
}

# List all halide structures
structure_list = []
for s in halide_structures:
    structure_list.append({
        "crystal_id": s["crystal_id"],
        "formula": s["formula"],
        "metals": s.get("metals", []),
        "halides": s["halides"],
        "template_type": s["template_type"],
        "band_gap_estimate_eV": s["band_gap_estimate_eV"],
        "chgnet_energy_per_atom_eV": s["chgnet_energy_per_atom_eV"],
        "n_atoms": s["n_atoms"]
    })

# Sort by crystal_id for consistency
structure_list.sort(key=lambda x: x["crystal_id"])

# Build result
result = {
    "total_halide_structures": total_halide,
    "halide_type_breakdown": {
        halide: {
            "count": count,
            "percentage": round(count / total_halide * 100, 2)
        }
        for halide, count in sorted(halide_counts.items())
    },
    "metal_type_breakdown": dict(sorted(metal_counts.items(), key=lambda x: -x[1])),
    "template_type_breakdown": dict(sorted(template_counts.items(), key=lambda x: -x[1])),
    "band_gap_statistics_eV": {k: round(v, 4) for k, v in bg_stats.items()},
    "chgnet_energy_statistics_eV_per_atom": {k: round(v, 4) for k, v in chg_stats.items()},
    "halide_structures": structure_list
}

print(json.dumps(result, indent=2))
