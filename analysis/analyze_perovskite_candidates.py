import json
import os
from pathlib import Path

# Get the mattergen directory
MATTERGEN_DIR = Path(__file__).parent

# Load the hybrid analysis data
data_path = MATTERGEN_DIR / "results_hybrid_8k_1000" / "balanced" / "balanced_analysis_hybrid.json"
with open(data_path) as f:
    data = json.load(f)

structures = data["balanced_structures"]

output_dir = MATTERGEN_DIR / ".." / "paper_results"
output_dir.mkdir(exist_ok=True)

# Metal and halide definitions
metals = {"Li","Na","K","Rb","Cs","Fr","Be","Mg","Ca","Sr","Ba","Ra",
          "Sc","Y","La","Ti","Zr","Hf","V","Nb","Ta","Cr","Mo","W",
          "Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir","Ni","Pd","Pt",
          "Cu","Ag","Au","Zn","Cd","Hg","Al","Ga","In","Tl","Si","Ge",
          "Sn","Pb","Bi","Sb","Ge","U","Th"}
halides = {"F","Cl","Br","I"}

print("="*80)
print("Perovskite-like structure analysis (metal + halide requirement)")
print("="*80)

perovskite_candidates = []

for s in structures:
    comp = s["composition"]
    metal_list = [e for e in comp if e in metals]
    halide_list = [e for e in comp if e in halides]
    
    has_metal = len(metal_list) > 0
    has_halide = len(halide_list) > 0
    
    if has_metal and has_halide:
        metal_count = sum(comp[m] for m in metal_list)
        halide_count = sum(comp[h] for h in halide_list)
        ratio = halide_count / metal_count if metal_count > 0 else 0
        
        # Check if ratio is close to 3 (ABX3 perovskite)
        is_perovskite_like = 2.0 <= ratio <= 4.0
        
        perovskite_candidates.append({
            "name": s["name"],
            "formula": s["formula"],
            "metals": metal_list,
            "halides": halide_list,
            "metal_count": metal_count,
            "halide_count": halide_count,
            "ratio": ratio,
            "is_perovskite_like": is_perovskite_like,
            "template_type": s.get("template_type", "Unknown"),
            "volume": s.get("volume", 0),
            "n_atoms": s.get("n_atoms", 0)
        })

print(f"\nTotal structures with both metal + halide: {len(perovskite_candidates)}")
print(f"Perovskite-like (ratio 2.0-4.0): {sum(1 for p in perovskite_candidates if p['is_perovskite_like'])}")
print()

# Sort by ratio closeness to 3.0
perovskite_candidates.sort(key=lambda x: abs(x["ratio"] - 3.0))

print("Top perovskite-like structures (sorted by ratio closeness to 3.0):")
print("-" * 80)
print(f"{'Name':<25} {'Formula':<30} {'Ratio':>8} {'Metals':<15} {'Halides':<10} {'Template':<20}")
print("-" * 80)

for p in perovskite_candidates[:20]:
    marker = "✓" if p["is_perovskite_like"] else "✗"
    print(f"{p['name']:<25} {p['formula']:<30} {p['ratio']:>8.2f} {','.join(p['metals']):<15} {','.join(p['halides']):<10} {p['template_type']:<20}")

# Save perovskite candidates
candidates_path = output_dir / "perovskite_candidates.json"
with open(candidates_path, "w") as f:
    json.dump(perovskite_candidates, f, indent=2)

print(f"\nSaved {len(perovskite_candidates)} candidates to {candidates_path}")

# Now select the top 5-10 for DFT validation
dft_selection = [p for p in perovskite_candidates if p["is_perovskite_like"]][:10]

print(f"\nSelected {len(dft_selection)} structures for DFT validation:")
for p in dft_selection:
    print(f"  - {p['name']}: {p['formula']} (ratio={p['ratio']:.2f})")

dft_path = output_dir / "dft_selection.json"
with open(dft_path, "w") as f:
    json.dump(dft_selection, f, indent=2)

print(f"\nSaved DFT selection to {dft_path}")
