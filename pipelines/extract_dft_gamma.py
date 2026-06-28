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

def extract_gap_gamma(path):
    """只提取Gamma点(k=0.0000 0.0000 0.0000)的带隙"""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # 精确匹配Gamma点
        if 'k = 0.0000 0.0000 0.0000' in line and 'bands (ev)' in line:
            i += 1  # 跳过空行
            
            # 读取能带能量
            energies = []
            while i < len(lines):
                stripped = lines[i].strip()
                if not stripped:
                    i += 1
                    continue
                if 'occupation' in stripped or 'k =' in stripped or 'End of band' in stripped:
                    break
                for v in stripped.split():
                    try:
                        energies.append(float(v))
                    except:
                        pass
                i += 1
            
            # 跳过空行到 occupation numbers
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # 读取 occupation numbers
            occs = []
            if i < len(lines) and 'occupation numbers' in lines[i]:
                i += 1
                while i < len(lines):
                    stripped = lines[i].strip()
                    if not stripped:
                        i += 1
                        continue
                    if 'k =' in stripped or 'End of band' in stripped:
                        break
                    for v in stripped.split():
                        try:
                            occs.append(float(v))
                        except:
                            pass
                    i += 1
            
            if len(energies) == len(occs) and len(energies) > 0:
                # 使用严格阈值: occ>=0.99为占据, occ<=0.01为未占据
                occ_energies = [e for e, o in zip(energies, occs) if o >= 0.99]
                unocc_energies = [e for e, o in zip(energies, occs) if o <= 0.01]
                
                if occ_energies and unocc_energies:
                    return min(unocc_energies) - max(occ_energies)
                
                # 如果严格阈值失败，使用0.5阈值
                occ_energies = [e for e, o in zip(energies, occs) if o > 0.5]
                unocc_energies = [e for e, o in zip(energies, occs) if o <= 0.5]
                
                if occ_energies and unocc_energies:
                    return min(unocc_energies) - max(occ_energies)
                
                return None
            return None
        
        i += 1
    
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

print("=== Gamma点带隙提取 (使用occ>=0.99/occ<=0.01严格阈值) ===")
print("| 编号 | 化学成分 | 金属 | 卤素 | 总能量(Ry) | SCF迭代 | Fermi(eV) | Gamma带隙(eV) | 类型 |")
print("|------|----------|------|------|-----------|---------|----------|--------------|------|")

for cid, formula, metal, halide in crystals:
    scf_path = f'{base}/crystal_{cid}_band/scf/crystal_{cid}_decoded.scf.out'
    nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
    
    scf = extract_scf_data(scf_path)
    gap = extract_gap_gamma(nscf_path)
    
    te = f"{scf['total_energy']:.2f}" if scf and scf['total_energy'] else "N/A"
    it = f"{scf['n_iter']}" if scf and scf['n_iter'] else "N/A"
    fe = f"{scf['fermi']:.2f}" if scf and scf['fermi'] else "N/A"
    gp = f"{gap:.2f}" if gap is not None else "N/A"
    
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
        'gap_gamma': gap,
        'type': etype
    }

# 保存到JSON
with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_corrected_gamma_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n结果已保存到: dft_corrected_gamma_results.json")

# 统计
n_sem = sum(1 for r in results.values() if r['type'] == '半导体')
n_met = sum(1 for r in results.values() if r['type'] == '金属')
n_na = sum(1 for r in results.values() if r['type'] == 'N/A')
print(f"统计: 半导体={n_sem}, 金属={n_met}, 未提取={n_na}")
