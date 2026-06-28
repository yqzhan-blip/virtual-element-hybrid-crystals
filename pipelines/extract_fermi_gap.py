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

def analyze_gap_fermi_based(kpoints, bands, occs, efermi):
    """基于Fermi能级的带隙分析
    
    VBM = 所有k点中 E < E_Fermi 的最高能级
    CBM = 所有k点中 E > E_Fermi 的最低能级
    
    直接带隙 = 同一k点的 CBM - VBM (取所有k点最小值)
    间接带隙 = 全局CBM - 全局VBM (可能不同k点)
    """
    if not kpoints or efermi is None:
        return None
    
    nk = len(kpoints)
    
    # 检查是否有部分占据态 (0.01 < occ < 0.99) - 这是smearing造成的伪金属
    partial_count = 0
    for ik in range(nk):
        for e, o in zip(bands[ik], occs[ik]):
            if 0.01 < o < 0.99:
                partial_count += 1
    
    # 基于Fermi能级的分割 (更准确的物理方法)
    vbm_candidates = []
    cbm_candidates = []
    
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        # VBM候选: E < E_Fermi 的能级 (不论occ值)
        vb_candidates = [e for e, o in zip(bands[ik], occs[ik]) if e < efermi]
        # CBM候选: E > E_Fermi 的能级
        cb_candidates = [e for e, o in zip(bands[ik], occs[ik]) if e > efermi]
        
        if vb_candidates:
            vbm_candidates.append((max(vb_candidates), ik))
        if cb_candidates:
            cbm_candidates.append((min(cb_candidates), ik))
    
    if not vbm_candidates or not cbm_candidates:
        return None
    
    global_vbm = max(vbm_candidates, key=lambda x: x[0])
    global_cbm = min(cbm_candidates, key=lambda x: x[0])
    
    indirect_gap = global_cbm[0] - global_vbm[0]
    
    # 直接带隙：计算每个k点的直接gap，取最小
    direct_gaps = []
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        vb = [e for e in bands[ik] if e < efermi]
        cb = [e for e in bands[ik] if e > efermi]
        
        if vb and cb:
            dg = min(cb) - max(vb)
            if dg > 0:
                direct_gaps.append((ik, dg))
    
    if not direct_gaps:
        return None
    
    min_direct = min(direct_gaps, key=lambda x: x[1])
    
    # 是否直接带隙：全局VBM和CBM是否在同一k点
    direct = (global_vbm[1] == global_cbm[1])
    
    # 如果全局VBM和CBM同k点，直接带隙就是间接带隙
    # 如果不同k点，直接带隙是最小直接带隙
    direct_gap_value = indirect_gap if direct else min_direct[1]
    
    return {
        'type': 'semiconductor' if indirect_gap > 0.1 else 'semimetal' if indirect_gap > 0 else 'metal',
        'gap': indirect_gap,  # 间接带隙
        'direct_gap': direct_gap_value,  # 直接带隙
        'direct': direct,
        'vbm_energy': global_vbm[0],
        'vbm_k': kpoints[global_vbm[1]],
        'vbm_k_index': global_vbm[1],
        'cbm_energy': global_cbm[0],
        'cbm_k': kpoints[global_cbm[1]],
        'cbm_k_index': global_cbm[1],
        'min_direct_k': kpoints[min_direct[0]],
        'min_direct_k_index': min_direct[0],
        'partial_occ_count': partial_count,
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
    
    print("| 编号 | 成分 | Fermi(eV) | 类型 | VBM(eV) | CBM(eV) | 间接带隙(eV) | 直接带隙? | 直接带隙(eV) | VBM_k | CBM_k | 部分占据 |")
    print("|------|------|-----------|------|---------|---------|-------------|----------|-------------|-------|-------|----------|")
    
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
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        result = analyze_gap_fermi_based(kpoints, bands, occs, efermi)
        
        efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
        
        if result:
            type_str = result['type']
            direct_str = "是" if result['direct'] else "否"
            vbm_k_str = f"({result['vbm_k'][0]:.2f},{result['vbm_k'][1]:.2f},{result['vbm_k'][2]:.2f})"
            cbm_k_str = f"({result['cbm_k'][0]:.2f},{result['cbm_k'][1]:.2f},{result['cbm_k'][2]:.2f})"
            
            print(f"| {cid} | {formula} | {efermi_str} | {type_str} | {result['vbm_energy']:.2f} | {result['cbm_energy']:.2f} | {result['gap']:.2f} | {direct_str} | {result['direct_gap']:.2f} | {vbm_k_str} | {cbm_k_str} | {result['partial_occ_count']} |")
            
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
                'type': result['type'],
                'partial_occ_count': result['partial_occ_count'],
            }
        else:
            print(f"| {cid} | {formula} | {efermi_str} | 无法解析 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    
    # 统计
    n_metal = sum(1 for r in all_results.values() if r['type'] == 'metal')
    n_sem = sum(1 for r in all_results.values() if r['type'] == 'semiconductor')
    n_semimetal = sum(1 for r in all_results.values() if r['type'] == 'semimetal')
    
    print(f"\n统计: 半导体={n_sem}, 半金属={n_semimetal}, 金属={n_metal}")
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_fermi_gap_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_fermi_gap_analysis.json")

if __name__ == '__main__':
    main()
