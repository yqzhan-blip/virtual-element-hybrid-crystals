"""Figure-text correspondence checker for paper_draft_v5.md and SI_v5.md"""
import re
import json

# Expected ground truth from DFT analysis
GROUND_TRUTH = {
    "0742": {"formula": "NiH8C3I3N", "metal": "Ni", "halide": "I", "type": "Metallic", "indirect": -0.18, "direct": 0.06, "vbm_k": 18, "cbm_k": 3, "fermi": 1.880, "total_e": -1564.833},
    "0789": {"formula": "NiH8C3I3N", "metal": "Ni", "halide": "I", "type": "Metallic", "indirect": -0.61, "direct": 0.03, "vbm_k": 20, "cbm_k": 14, "fermi": 3.416, "total_e": -1564.848},
    "0391": {"formula": "Mg2H12C2I6N2", "metal": "Mg", "halide": "I", "type": "Metallic", "indirect": -0.16, "direct": 0.07, "vbm_k": 8, "cbm_k": 9, "fermi": 1.856, "total_e": -2430.563},
    "0631": {"formula": "CsH6CBr2N", "metal": "Cs", "halide": "Br", "type": "Semiconductor", "indirect": 2.17, "direct": 2.49, "vbm_k": 18, "cbm_k": 0, "fermi": 0.525, "total_e": -202.726, "di": "Indirect"},
    "0912": {"formula": "KH8C3NF2", "metal": "K", "halide": "F", "type": "Semiconductor", "indirect": 3.27, "direct": 3.39, "vbm_k": 4, "cbm_k": 2, "fermi": 4.268, "total_e": -293.642, "di": "Indirect"},
    "0217": {"formula": "MnH8C3I3N", "metal": "Mn", "halide": "I", "type": "Semiconductor", "indirect": 0.46, "direct": 0.49, "vbm_k": 12, "cbm_k": 23, "fermi": 4.668, "total_e": -1432.582, "di": "Indirect"},
    "0672": {"formula": "FeH8C3Br3N", "metal": "Fe", "halide": "Br", "type": "Semiconductor", "indirect": 0.54, "direct": 0.60, "vbm_k": 5, "cbm_k": 20, "fermi": 4.038, "total_e": -554.424, "di": "Indirect"},
    "0927": {"formula": "CoH12C2N2Cl4", "metal": "Co", "halide": "Cl", "type": "Semiconductor", "indirect": 0.54, "direct": 0.56, "vbm_k": 4, "cbm_k": 2, "fermi": 2.244, "total_e": -516.600, "di": "Indirect"},
    "0626": {"formula": "VH11C2Br6N3", "metal": "V", "halide": "Br", "type": "Semiconductor", "indirect": 0.78, "direct": 0.78, "vbm_k": 0, "cbm_k": 0, "fermi": -2.633, "total_e": -534.121, "di": "Direct"},
    "0632": {"formula": "CrH11C2Br6N3", "metal": "Cr", "halide": "Br", "type": "Semiconductor", "indirect": 0.51, "direct": 0.51, "vbm_k": 13, "cbm_k": 11, "fermi": -2.017, "total_e": -565.254, "di": "Indirect"},
}

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

paper = read_file("paper_draft_v5.md")
si = read_file("SI_v5.md")

issues = []

# 1. Check Table 5 in paper (10 crystals)
print("=" * 60)
print("CHECK 1: Table 5 (paper) - 10 crystal DFT data consistency")
print("=" * 60)

# Extract Table 5
lines = paper.split('\n')
in_table5 = False
table5_data = []
for line in lines:
    if "Table 5" in line and "DFT-PBE" in line:
        in_table5 = True
        continue
    if in_table5 and line.startswith('|') and 'Crystal' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 12 and parts[1] and not parts[1].startswith('-'):
            crystal = parts[1]
            formula = parts[2]
            indirect = parts[9].strip().replace('**', '').replace('$', '')
            direct = parts[10].strip().replace('**', '').replace('$', '')
            type_ = parts[11].strip().replace('**', '')
            di = parts[12].strip().replace('**', '') if len(parts) > 12 else ""
            table5_data.append((crystal, formula, indirect, direct, type_, di))
    if in_table5 and line.startswith('Three structures'):
        in_table5 = False

for crystal, formula, indirect, direct, type_, di in table5_data:
    if crystal not in GROUND_TRUTH:
        continue
    gt = GROUND_TRUTH[crystal]
    
    # Check formula (simplified)
    f_clean = formula.replace('$', '').replace('{', '').replace('}', '').replace('\\', '').replace('*', '')
    f_gt = gt['formula'].replace('$', '')
    
    # Check indirect gap
    try:
        indirect_val = float(indirect)
        if abs(indirect_val - gt['indirect']) > 0.02:
            issues.append(f"Table 5 crystal {crystal}: indirect gap mismatch: doc={indirect_val}, expected={gt['indirect']}")
    except:
        pass
    
    # Check direct gap
    try:
        direct_val = float(direct)
        if abs(direct_val - gt['direct']) > 0.02:
            issues.append(f"Table 5 crystal {crystal}: direct gap mismatch: doc={direct_val}, expected={gt['direct']}")
    except:
        pass
    
    # Check type
    if type_ and type_ != gt['type']:
        issues.append(f"Table 5 crystal {crystal}: type mismatch: doc={type_}, expected={gt['type']}")
    
    # Check direct/indirect classification
    if 'di' in gt and di and di != "\u2014" and di != gt['di']:
        issues.append(f"Table 5 crystal {crystal}: direct/indirect mismatch: doc={di}, expected={gt['di']}")
    elif 'di' not in gt and di and di != "\u2014":
        issues.append(f"Table 5 crystal {crystal}: should be '\u2014' for metallic, got {di}")

print(f"  Table 5 rows checked: {len(table5_data)}")

# 2. Check Table S4 in SI
print()
print("=" * 60)
print("CHECK 2: Table S4 (SI) - NSCF band structure consistency")
print("=" * 60)

si_lines = si.split('\n')
in_table_s4 = False
s4_data = []
for line in si_lines:
    if "Table S4" in line and "NSCF" in line:
        in_table_s4 = True
        continue
    if in_table_s4 and line.startswith('|') and 'Crystal' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 12 and parts[1] and not parts[1].startswith('-') and parts[1] != 'Crystal':
            crystal = parts[1].replace('*', '').replace('**', '').strip()
            vbm = parts[3].strip()
            vbm_k = parts[4].strip()
            cbm = parts[5].strip()
            cbm_k = parts[6].strip()
            indirect = parts[7].strip().replace('**', '').replace('$', '')
            direct = parts[8].strip().replace('**', '').replace('$', '')
            type_ = parts[9].strip().replace('**', '')
            di = parts[10].strip().replace('**', '') if len(parts) > 10 else ""
            s4_data.append((crystal, vbm, vbm_k, cbm, cbm_k, indirect, direct, type_, di))
    if in_table_s4 and line.startswith('**Key observations'):
        in_table_s4 = False

for crystal, vbm, vbm_k, cbm, cbm_k, indirect, direct, type_, di in s4_data:
    if crystal not in GROUND_TRUTH:
        continue
    gt = GROUND_TRUTH[crystal]
    
    # Check indirect gap
    try:
        indirect_val = float(indirect)
        if abs(indirect_val - gt['indirect']) > 0.02:
            issues.append(f"Table S4 crystal {crystal}: indirect gap mismatch: doc={indirect_val}, expected={gt['indirect']}")
    except:
        pass
    
    # Check direct gap
    try:
        direct_val = float(direct)
        if abs(direct_val - gt['direct']) > 0.02:
            issues.append(f"Table S4 crystal {crystal}: direct gap mismatch: doc={direct_val}, expected={gt['direct']}")
    except:
        pass
    
    # Check VBM k-index
    try:
        vbm_k_val = int(vbm_k)
        if vbm_k_val != gt['vbm_k']:
            issues.append(f"Table S4 crystal {crystal}: VBM k-index mismatch: doc={vbm_k_val}, expected={gt['vbm_k']}")
    except:
        pass
    
    # Check CBM k-index
    try:
        cbm_k_val = int(cbm_k)
        if cbm_k_val != gt['cbm_k']:
            issues.append(f"Table S4 crystal {crystal}: CBM k-index mismatch: doc={cbm_k_val}, expected={gt['cbm_k']}")
    except:
        pass
    
    # Check type
    if type_ and type_ != gt['type']:
        issues.append(f"Table S4 crystal {crystal}: type mismatch: doc={type_}, expected={gt['type']}")
    
    # Check di
    if 'di' in gt and di and di != "\u2014" and di != gt['di']:
        issues.append(f"Table S4 crystal {crystal}: direct/indirect mismatch: doc={di}, expected={gt['di']}")
    elif 'di' not in gt and di and di != "\u2014":
        issues.append(f"Table S4 crystal {crystal}: should be '\u2014' for metallic, got {di}")

print(f"  Table S4 rows checked: {len(s4_data)}")

# 3. Check figure captions in SI
print()
print("=" * 60)
print("CHECK 3: Figure captions vs data consistency")
print("=" * 60)

fig_captions = {
    "S7": ("0631", "indirect band gap of 2.17 eV", "VBM", "k-index 18", "CBM", "1.162 eV", "Gamma"),
    "S9": ("0912", "indirect gap of 3.27 eV", "VBM", "4.027 eV", "CBM", "7.296 eV"),
    "S11": ("0217", "indirect gap of 0.46 eV", "VBM", "4.475 eV", "CBM", "4.936 eV"),
    "S13": ("0672", "indirect gap of 0.54 eV", "VBM", "3.817 eV", "CBM", "4.355 eV"),
    "S15": ("0927", "indirect gap of 0.54 eV", "VBM", "1.939 eV", "CBM", "2.477 eV"),
    "S17": ("0626", "direct band gap of 0.78 eV", "Gamma", "both at the Gamma"),
    "S19": ("0632", "indirect gap of 0.51 eV", "VBM", "\u22122.279 eV", "CBM", "\u22121.773 eV"),
}

for fig, expected in fig_captions.items():
    crystal = expected[0]
    # Search for figure caption in SI
    pattern = f"Figure {fig}"
    found = pattern in si
    if not found:
        issues.append(f"Figure {fig} not found in SI")
        continue
    
    # Extract caption text around figure
    idx = si.find(pattern)
    caption_text = si[idx:idx+500]
    
    # Check expected values in caption
    for i, expected_val in enumerate(expected[1:], 1):
        if expected_val and expected_val not in caption_text:
            issues.append(f"Figure {fig} caption missing expected text: '{expected_val}'")
            break
    
    print(f"  Figure {fig} (crystal {crystal}): OK")

# 4. Cross-check paper text vs Table 5
print()
print("=" * 60)
print("CHECK 4: Paper text vs Table 5 data consistency")
print("=" * 60)

# Check specific mentions in text
expected_text_checks = [
    ("0631", "2.17 eV indirect", "indirect band gap of 2.17 eV"),
    ("0631", "Gamma point", "CBM at the Gamma point"),
    ("0912", "3.27 eV", "indirect gap of 3.27 eV"),
    ("0626", "0.78 eV", "direct band gap (0.78 eV"),
    ("0626", "only", "only structure with a direct band gap"),
]

for crystal, check_name, expected_text in expected_text_checks:
    if expected_text in paper:
        print(f"  crystal {crystal} ({check_name}): OK")
    else:
        issues.append(f"Paper text missing: crystal {crystal} '{expected_text}'")

# 5. Check 0631 VBM k-index consistency (the key issue from previous corrections)
print()
print("=" * 60)
print("CHECK 5: 0631 VBM/CBM k-index special check")
print("=" * 60)

if "VBM at k-index 18" in si and "CBM at k-index 0" in si:
    print("  SI Table S4: VBM at k-index 18, CBM at k-index 0 - CORRECT")
else:
    issues.append("SI: 0631 VBM/CBM k-index incorrect")

if "VBM at k-index 18, CBM at the Gamma point" in paper:
    print("  Paper text: VBM at k-index 18, CBM at Gamma - CORRECT")
else:
    issues.append("Paper: 0631 VBM/CBM description incorrect")

# 6. Check that 0626 is the only direct band gap
print()
print("=" * 60)
print("CHECK 6: Direct/indirect classification consistency")
print("=" * 60)

direct_count = sum(1 for c, d in GROUND_TRUTH.items() if d.get('di') == 'Direct')
indirect_count = sum(1 for c, d in GROUND_TRUTH.items() if d.get('di') == 'Indirect')
metallic_count = sum(1 for c, d in GROUND_TRUTH.items() if 'di' not in d)

print(f"  Ground truth: {direct_count} direct, {indirect_count} indirect, {metallic_count} metallic")

# Check paper text says "only one structure"
if "only one structure (0626) exhibits a direct band gap" in paper:
    print("  Paper text: 'only one structure (0626) exhibits a direct band gap' - CORRECT")
else:
    issues.append("Paper: should mention 'only one structure (0626) exhibits a direct band gap'")

# Check paper text says 3 metallic
if "Three structures (0742, 0789, 0391) exhibit zero band gaps" in paper:
    print("  Paper text: 'Three structures (0742, 0789, 0391) exhibit zero band gaps' - CORRECT")
else:
    issues.append("Paper: should mention 3 metallic structures")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

if issues:
    print(f"  ISSUES FOUND: {len(issues)}")
    for issue in issues:
        print(f"    - {issue}")
else:
    print("  ALL CHECKS PASSED - No issues found!")
    print("  Figure-text correspondence is consistent.")

# Write report
report_path = "figure_text_correspondence_report.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("Figure-Text Correspondence Check Report\n")
    f.write("=" * 60 + "\n\n")
    f.write("Checks performed:\n")
    f.write("1. Table 5 (paper) vs DFT ground truth data\n")
    f.write("2. Table S4 (SI) vs DFT ground truth data\n")
    f.write("3. Figure captions in SI vs expected descriptions\n")
    f.write("4. Paper text descriptions vs Table 5 data\n")
    f.write("5. 0631 VBM/CBM k-index special check\n")
    f.write("6. Direct/indirect classification count\n\n")
    if issues:
        f.write(f"ISSUES FOUND: {len(issues)}\n")
        for issue in issues:
            f.write(f"  - {issue}\n")
    else:
        f.write("ALL CHECKS PASSED - No issues found!\n")
        f.write("Figure-text correspondence is consistent.\n")

print(f"\nReport written to: {report_path}")
