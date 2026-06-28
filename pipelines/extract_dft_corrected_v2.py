import re, os, json

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
        lines = f.readlines()
    
    k_points = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 匹配 k = 0.0000 0.0000 0.0000 (  4417 PWs)   bands (ev):
        if 'k =' in line and 'bands (ev)' in line:
            i += 1  # 跳过空行
            
            # 读取能带能量 (多行，每行最多8个浮点数)
            energies = []
            while i < len(lines) and lines[i].strip() and 'occupation' not in lines[i] and 'k =' not in lines[i]:
                vals = lines[i].strip().split()
                for v in vals:
                    try:
                        energies.append(float(v))
                    except:
                        pass
                i += 1
            
            # 跳过空行到 occupation numbers
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # 读取 occupation numbers (同样格式)
            occs = []
            if i < len(lines) and 'occupation numbers' in lines[i]:
                i += 1
                while i < len(lines) and lines[i].strip() and 'k =' not in lines[i]:
                    vals = lines[i].strip().split()
                    for v in vals:
                        try:
                            occs.append(float(v))
                        except:
                            pass
                    i += 1
            
            if len(energies) == len(occs) and len(energies) > 0:
                k_points.append(list(zip(energies, occs)))
        
        i += 1
    
    if not k_points:
        return None
    
    gaps = []
    for kp in k_points:
        occ_energies = [e for e, o in kp if o > 0.5]
        unocc_energies = [e for e, o in kp if o <= 0.5]
        if occ_energies and unocc_energies:
            gaps.append(min(unocc_energies) - max(occ_energies))
    
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

results = {}

print("| 编号 | 化学成分 | 金属 | 卤素 | 总能量(Ry) | SCF迭代 | Fermi(eV) | 带隙(eV) | 类型 |")
print("|------|----------|------|------|-----------|---------|----------|---------|------|")

for cid, formula, metal, halide in crystals:
    scf_path = f'{base}/crystal_{cid}_band/scf/crystal_{cid}_decoded.scf.out'
    nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
    
    scf = extract_scf_data(scf_path)
    gap = extract_gap(nscf_path)
    
    te = f"{scf['total_energy']:.2f}" if scf and scf['total_energy'] else "N/A"
    it = f"{scf['n_iter']}" if scf and scf['n_iter'] else "N/A"
    fe = f"{scf['fermi']:.2f}" if scf and scf['fermi'] else "N/A"
    gp = f"{gap:.2f}" if gap is not None else "N/A"
    
    # 判断类型
    if gap is None:
        etype = "N/A"
    elif gap > 0.1:
        etype = "半导体"
    else:
        etype = "金属"
    
    print(f"| {cid} | {formula} | {metal} | {halide} | {te} | {it} | {fe} | {gp} | {etype} |")
    
    results[cid] = {
        'formula': formula,
        'metal': metal,
        'halide': halide,
        'total_energy': scf['total_energy'] if scf else None,
        'n_iter': scf['n_iter'] if scf else None,
        'fermi': scf['fermi'] if scf else None,
        'gap': gap,
        'type': etype
    }

# 保存到JSON
with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_corrected_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n结果已保存到: dft_corrected_results.json")
