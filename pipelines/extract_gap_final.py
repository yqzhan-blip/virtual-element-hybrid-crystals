import re, os, json, sys
import numpy as np

def parse_nscf_all_kpoints(path):
    """从NSCF输出解析所有k点的能带和占据数"""
    with open(path, 'r') as f:
        text = f.read()
    
    pattern = r'k\s*=\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+\(\s*\d+\s+PW[s]?\)\s+bands \(ev\):'
    matches = list(re.finditer(pattern, text))
    
    kpoints = []
    bands_blocks = []
    occ_blocks = []
    
    for i, m in enumerate(matches):
        kx, ky, kz = map(float, m.groups())
        kpoints.append((kx, ky, kz))
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end]
        
        lines = block.split('\n')
        energies = []
        occs = []
        in_occ = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if 'occupation numbers' in stripped:
                in_occ = True
                continue
            if 'k =' in stripped or 'End of band' in stripped:
                break
            
            if not in_occ:
                for v in stripped.split():
                    try:
                        energies.append(float(v))
                    except:
                        pass
            else:
                for v in stripped.split():
                    try:
                        occs.append(float(v))
                    except:
                        pass
        
        bands_blocks.append(energies)
        occ_blocks.append(occs)
    
    return kpoints, bands_blocks, occ_blocks

def analyze_gap_homo_lumo(kpoints, bands, occs):
    """使用HOMO/LUMO方法分析带隙
    
    HOMO: 每个k点最高能量的占据态 (occ > 0.5)
    LUMO: 每个k点最低能量的空态 (occ < 0.5)
    
    直接带隙: 同一k点的LUMO - HOMO (取所有k点最小值)
    间接带隙: 全局最低LUMO - 全局最高HOMO
    """
    if not kpoints:
        return None
    
    nk = len(kpoints)
    direct_gaps = []  # (k_index, gap, homo, lumo)
    
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        # HOMO: 最高能量的占据态 (occ > 0.5)
        homo_candidates = [(e, occ) for e, occ in zip(bands[ik], occs[ik]) if occ > 0.5]
        # LUMO: 最低能量的空态 (occ < 0.5)
        lumo_candidates = [(e, occ) for e, occ in zip(bands[ik], occs[ik]) if occ < 0.5]
        
        if not homo_candidates or not lumo_candidates:
            continue
        
        homo = max(homo_candidates, key=lambda x: x[0])  # 最高能量
        lumo = min(lumo_candidates, key=lambda x: x[0])  # 最低能量
        
        gap = lumo[0] - homo[0]
        if gap > 0:  # 正带隙
            direct_gaps.append((ik, gap, homo[0], lumo[0], homo[1], lumo[1]))
    
    if not direct_gaps:
        return None
    
    # 最小直接带隙
    min_gap_info = min(direct_gaps, key=lambda x: x[1])
    min_ik, min_gap, min_homo, min_lumo, min_homo_occ, min_lumo_occ = min_gap_info
    
    # 全局HOMO和LUMO
    all_homo = [(h, ik) for _, _, h, _, _, _ in direct_gaps]
    all_lumo = [(l, ik) for _, _, _, l, _, _ in direct_gaps]
    
    global_homo = max(all_homo, key=lambda x: x[0])  # 所有k点中最高HOMO
    global_lumo = min(all_lumo, key=lambda x: x[0])  # 所有k点中最低LUMO
    
    indirect_gap = global_lumo[0] - global_homo[0]
    
    # 是否直接带隙：全局HOMO和全局LUMO是否在同一k点
    direct = (global_homo[1] == global_lumo[1])
    
    # 最小直接带隙值
    min_direct_gap = min_gap
    
    return {
        'gap': indirect_gap,  # 间接带隙 (全局LUMO - 全局HOMO)
        'direct_gap': min_direct_gap,  # 最小直接带隙
        'direct': direct,  # 是否直接带隙 (全局HOMO和LUMO同k点)
        'homo_energy': global_homo[0],  # VBM = 全局最高HOMO
        'homo_k': kpoints[global_homo[1]],
        'homo_k_index': global_homo[1],
        'lumo_energy': global_lumo[0],  # CBM = 全局最低LUMO
        'lumo_k': kpoints[global_lumo[1]],
        'lumo_k_index': global_lumo[1],
        'min_direct_k': kpoints[min_ik],
        'min_direct_k_index': min_ik,
        'min_direct_homo': min_homo,
        'min_direct_lumo': min_lumo,
    }

def main():
    base = 'C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs'
    
    crystals = [
        ('0742', 'NiH8C3I3N'),
        ('0789', 'NiH8C3I3N'),
        ('0391', 'Mg2H12C2I6N2'),
        ('0631', 'CsH6CBr2N'),
        ('0912', 'KH8C3NF2'),
        ('0217', 'MnH8C3I3N'),
        ('0672', 'FeH8C3Br3N'),
        ('0927', 'CoH12C2N2Cl4'),
        ('0626', 'VH11C2Br6N3'),
        ('0632', 'CrH11C2Br6N3'),
    ]
    
    print("| 编号 | 成分 | Fermi(eV) | HOMO/VBM(eV) | LUMO/CBM(eV) | 间接带隙(eV) | 直接带隙? | 最小直接带隙(eV) | HOMO_k | LUMO_k |")
    print("|------|------|-----------|-------------|-------------|-------------|----------|----------------|--------|--------|")
    
    all_results = {}
    
    for cid, formula in crystals:
        scf_path = f'{base}/crystal_{cid}_band/scf/crystal_{cid}_decoded.scf.out'
        nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
        
        # 提取Fermi能级
        efermi = None
        if os.path.exists(scf_path):
            with open(scf_path, 'r') as f:
                for line in f:
                    m = re.search(r'the Fermi energy is\s+([-\d.]+)\s*ev', line, re.I)
                    if m:
                        efermi = float(m.group(1))
                        break
        
        # 解析NSCF
        kpoints, bands, occs = parse_nscf_all_kpoints(nscf_path)
        
        if not kpoints:
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        gap_info = analyze_gap_homo_lumo(kpoints, bands, occs)
        
        if gap_info:
            direct_str = "是" if gap_info['direct'] else "否"
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            homo_k_str = f"({gap_info['homo_k'][0]:.2f},{gap_info['homo_k'][1]:.2f},{gap_info['homo_k'][2]:.2f})"
            lumo_k_str = f"({gap_info['lumo_k'][0]:.2f},{gap_info['lumo_k'][1]:.2f},{gap_info['lumo_k'][2]:.2f})"
            
            print(f"| {cid} | {formula} | {efermi_str} | {gap_info['homo_energy']:.2f} | {gap_info['lumo_energy']:.2f} | {gap_info['gap']:.2f} | {direct_str} | {gap_info['direct_gap']:.2f} | {homo_k_str} | {lumo_k_str} |")
            
            all_results[cid] = {
                'formula': formula,
                'fermi': efermi,
                'homo': gap_info['homo_energy'],
                'homo_k': gap_info['homo_k'],
                'homo_k_index': gap_info['homo_k_index'],
                'lumo': gap_info['lumo_energy'],
                'lumo_k': gap_info['lumo_k'],
                'lumo_k_index': gap_info['lumo_k_index'],
                'gap': gap_info['gap'],
                'direct': gap_info['direct'],
                'direct_gap': gap_info['direct_gap'],
                'min_direct_k': gap_info['min_direct_k'],
                'min_direct_k_index': gap_info['min_direct_k_index'],
                'min_direct_homo': gap_info['min_direct_homo'],
                'min_direct_lumo': gap_info['min_direct_lumo'],
            }
        else:
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_gap_analysis_final.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_gap_analysis_final.json")

if __name__ == '__main__':
    main()
