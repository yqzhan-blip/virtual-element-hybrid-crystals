import re, glob, os

def extract_scf_data(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        text = f.read()
    
    te_match = re.findall(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', text)
    total_energy = float(te_match[-1]) if te_match else None
    
    iter_match = re.findall(r'iteration #\s*(\d+)', text)
    n_iter = int(iter_match[-1]) if iter_match else None
    
    fermi_match = re.findall(r'the Fermi energy is\s+([-\d.]+)\s*ev', text, re.I)
    fermi = float(fermi_match[-1]) if fermi_match else None
    
    return {'total_energy': total_energy, 'n_iter': n_iter, 'fermi': fermi}

def extract_gap(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        text = f.read()
    
    k_blocks = []
    
    pattern1 = r'k\s*=\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s*\n)+)'
    matches1 = re.findall(pattern1, text)
    for m in matches1:
        kx, ky, kz, block = m
        lines = block.strip().split('\n')
        bands = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    bands.append((int(parts[0]), float(parts[1]), float(parts[2])))
                except:
                    pass
        if bands:
            k_blocks.append(bands)
    
    pattern2 = r'k\(\s*\d+\)\s*=\s*\(\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\s*\)\s*\(.*?\)\s*\n((?:\s*\d+\s+[-\d.]+\s+[-\d.]+\s*\n)+)'
    matches2 = re.findall(pattern2, text)
    for block in matches2:
        lines = block.strip().split('\n')
        bands = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    bands.append((int(parts[0]), float(parts[1]), float(parts[2])))
                except:
                    pass
        if bands:
            k_blocks.append(bands)
    
    if not k_blocks:
        return None
    
    gaps = []
    for bands in k_blocks:
        occ = [e for _, e, o in bands if o > 0.5]
        unocc = [e for _, e, o in bands if o <= 0.5]
        if occ and unocc:
            gaps.append(min(unocc) - max(occ))
    
    if gaps:
        return min(gaps)
    return None

base = 'C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs'

crystals = [
    ('0742', 'NiH8C3I3N', 'Ni', 'I'),
    ('0789', 'NiH8C3I3N', 'Ni', 'I'),
    ('0391', 'Mg2H12C2I6N2', 'Mg', 'I'),
    ('0631', 'CsH6CBr2N', 'Cs', 'Br'),
    ('0912', 'KH8C3NF2', 'K', 'F'),
    ('0217', 'MnH8C3I3N', 'Mn', 'I'),
    ('0672', 'FeH8C3Br3N', 'Fe', 'Br'),
    ('0927', 'CoH12C2N2Cl4', 'Co', 'Cl'),
    ('0626', 'VH11C2Br6N3', 'V', 'Br'),
    ('0632', 'CrH11C2Br6N3', 'Cr', 'Br'),
]

print("| 编号 | 化学成分 | 金属 | 卤素 | 总能量(Ry) | SCF迭代 | Fermi(eV) | 带隙(eV) |")
print("|------|----------|------|------|-----------|---------|----------|---------|")

for cid, formula, metal, halide in crystals:
    scf_path = f'{base}/crystal_{cid}_band/scf/crystal_{cid}_decoded.scf.out'
    nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
    
    scf = extract_scf_data(scf_path)
    gap = extract_gap(nscf_path)
    
    te = f"{scf['total_energy']:.2f}" if scf and scf['total_energy'] else "N/A"
    it = f"{scf['n_iter']}" if scf and scf['n_iter'] else "N/A"
    fe = f"{scf['fermi']:.2f}" if scf and scf['fermi'] else "N/A"
    gp = f"{gap:.2f}" if gap is not None else "N/A"
    
    print(f"| {cid} | {formula} | {metal} | {halide} | {te} | {it} | {fe} | {gp} |")

