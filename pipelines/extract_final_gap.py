import re, os, json
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

def analyze_gap_strict(kpoints, bands, occs):
    """使用严格占据阈值(0.99/0.01)分析带隙
    
    适用于smearing计算的NSCF输出，其中occ可能为分数值。
    
    直接带隙 = 同一k点的CBM - VBM (取所有k点最小值)
    间接带隙 = 全局最低CBM - 全局最高VBM
    """
    if not kpoints:
        return None
    
    nk = len(kpoints)
    direct_gaps = []
    
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        # 严格占据: occ >= 0.99 (完全占据)
        occ_energies = [e for e, o in zip(bands[ik], occs[ik]) if o >= 0.99]
        # 严格空态: occ <= 0.01 (完全空)
        empty_energies = [e for e, o in zip(bands[ik], occs[ik]) if o <= 0.01]
        
        if not occ_energies or not empty_energies:
            continue
        
        vbm_k = max(occ_energies)
        cbm_k = min(empty_energies)
        gap = cbm_k - vbm_k
        
        if gap > 0:
            direct_gaps.append((ik, gap, vbm_k, cbm_k))
    
    if not direct_gaps:
        return None
    
    # 最小直接带隙
    min_direct = min(direct_gaps, key=lambda x: x[1])
    
    # 间接带隙: 全局最高VBM和全局最低CBM
    all_vbm = [(vbm, ik) for _, _, vbm, _ in direct_gaps]
    all_cbm = [(cbm, ik) for _, _, _, cbm in direct_gaps]
    
    global_vbm = max(all_vbm, key=lambda x: x[0])
    global_cbm = min(all_cbm, key=lambda x: x[0])
    
    indirect_gap = global_cbm[0] - global_vbm[0]
    
    # 是否直接带隙: 全局VBM和CBM是否在同一k点
    direct = (global_vbm[1] == global_cbm[1])
    
    return {
        'gap': indirect_gap,
        'direct_gap': min_direct[1],
        'direct': direct,
        'vbm_energy': global_vbm[0],
        'vbm_k': kpoints[global_vbm[1]],
        'vbm_k_index': global_vbm[1],
        'cbm_energy': global_cbm[0],
        'cbm_k': kpoints[global_cbm[1]],
        'cbm_k_index': global_cbm[1],
        'min_direct_k': kpoints[min_direct[0]],
        'min_direct_k_index': min_direct[0],
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
    
    print("| 编号 | 成分 | Fermi(eV) | VBM(eV) | CBM(eV) | 间接带隙(eV) | 直接带隙? | 最小直接带隙(eV) | VBM_k_index | CBM_k_index |")
    print("|------|------|-----------|---------|---------|-------------|----------|----------------|------------|------------|")
    
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
        
        result = analyze_gap_strict(kpoints, bands, occs)
        
        if result:
            direct_str = "是" if result['direct'] else "否"
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            
            print(f"| {cid} | {formula} | {efermi_str} | {result['vbm_energy']:.2f} | {result['cbm_energy']:.2f} | {result['gap']:.2f} | {direct_str} | {result['direct_gap']:.2f} | {result['vbm_k_index']} | {result['cbm_k_index']} |")
            
            all_results[cid] = {
                'formula': formula,
                'fermi': efermi,
                'vbm': result['vbm_energy'],
                'vbm_k': result['vbm_k'],
                'vbm_k_index': result['vbm_k_index'],
                'cbm': result['cbm_energy'],
                'cbm_k': result['cbm_k'],
                'cbm_k_index': result['cbm_k_index'],
                'gap': result['gap'],
                'direct': result['direct'],
                'direct_gap': result['direct_gap'],
                'min_direct_k': result['min_direct_k'],
                'min_direct_k_index': result['min_direct_k_index'],
            }
        else:
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_final_gap.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_final_gap.json")

if __name__ == '__main__':
    main()
