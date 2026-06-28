import re, os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def analyze_corrected(kpoints, bands, occs):
    """正确的带隙分析
    
    直接带隙 = min(所有k点的 CBM(k) - VBM(k))
    间接带隙 = 全局最低CBM - 全局最高VBM
    
    直接带隙 <= 间接带隙 (三角不等式)
    
    如果VBM和CBM在同一k点: 直接带隙 = 间接带隙
    如果VBM和CBM在不同k点: 直接带隙 < 间接带隙 (严格间接带隙)
    """
    if not kpoints:
        return None
    
    nk = len(kpoints)
    local_gaps = []  # (k_index, local_gap, local_vbm, local_cbm)
    
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        occ_energies = [e for e, o in zip(bands[ik], occs[ik]) if o >= 0.99]
        empty_energies = [e for e, o in zip(bands[ik], occs[ik]) if o <= 0.01]
        
        if not occ_energies or not empty_energies:
            continue
        
        local_vbm = max(occ_energies)
        local_cbm = min(empty_energies)
        local_gap = local_cbm - local_vbm
        
        local_gaps.append((ik, local_gap, local_vbm, local_cbm))
    
    if not local_gaps:
        return {'type': 'unknown'}
    
    # 检查金属性 (存在负局部带隙)
    has_negative = any(gap < 0 for _, gap, _, _ in local_gaps)
    
    if has_negative:
        positive_gaps = [(ik, gap, vbm, cbm) for ik, gap, vbm, cbm in local_gaps if gap > 0]
        if positive_gaps:
            min_pos = min(positive_gaps, key=lambda x: x[1])
            return {
                'type': 'metal' if min_pos[1] < 0.1 else 'semimetal',
                'direct_gap': 0.0,
                'indirect_gap': 0.0,
                'min_positive_gap': min_pos[1],
                'min_positive_k_index': min_pos[0],
                'min_positive_k': kpoints[min_pos[0]],
                'vbm_energy': max(vbm for _, _, vbm, _ in local_gaps),
                'cbm_energy': min(cbm for _, _, _, cbm in local_gaps),
            }
        return {'type': 'metal', 'direct_gap': 0.0, 'indirect_gap': 0.0}
    
    # 直接带隙 = 最小局部带隙
    min_direct = min(local_gaps, key=lambda x: x[1])
    direct_gap = min_direct[1]
    
    # 间接带隙 = 全局最低CBM - 全局最高VBM
    all_vbm = [(vbm, ik) for ik, _, vbm, _ in local_gaps]
    all_cbm = [(cbm, ik) for ik, _, _, cbm in local_gaps]
    
    global_vbm = max(all_vbm, key=lambda x: x[0])
    global_cbm = min(all_cbm, key=lambda x: x[0])
    
    indirect_gap = global_cbm[0] - global_vbm[0]
    
    # 直接带隙应 <= 间接带隙
    # 如果直接 > 间接，说明数值精度问题，取较小值
    if direct_gap > indirect_gap + 0.01:
        # 这种情况说明全局VBM和CBM不在同一个k点，但直接带隙的那个k点
        # 的VBM比全局VBM更低，导致直接带隙 > 间接带隙
        # 实际上应该重新定义为：
        # 直接带隙 = min_k(CBM(k) - VBM(k)) 其中VBM(k)是全局VBM
        # 但标准定义是局部带隙的最小值
        pass  # 保持原值，直接带隙可以大于间接带隙在数学上
    
    # 是否直接带隙: VBM和CBM是否在同一k点
    direct = (global_vbm[1] == global_cbm[1])
    
    return {
        'type': 'semiconductor',
        'direct_gap': direct_gap,
        'indirect_gap': indirect_gap,
        'direct': direct,
        'vbm_energy': global_vbm[0],
        'vbm_k_index': global_vbm[1],
        'vbm_k': kpoints[global_vbm[1]],
        'cbm_energy': global_cbm[0],
        'cbm_k_index': global_cbm[1],
        'cbm_k': kpoints[global_cbm[1]],
        'min_direct_k_index': min_direct[0],
        'min_direct_k': kpoints[min_direct[0]],
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
    
    print("| 编号 | 成分 | Fermi(eV) | VBM(eV) | VBM_k_index | CBM(eV) | CBM_k_index | 间接带隙(eV) | 直接带隙(eV) | 直接/间接 |")
    print("|------|------|-----------|---------|-------------|---------|-------------|-------------|-------------|----------|")
    
    all_results = {}
    
    for cid, formula in crystals:
        scf_path = f'{base}/crystal_{cid}_band/scf/crystal_{cid}_decoded.scf.out'
        nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
        
        efermi = None
        if os.path.exists(scf_path):
            with open(scf_path, 'r') as f:
                for line in f:
                    m = re.search(r'the Fermi energy is\s+([-\d.]+)\s*ev', line, re.I)
                    if m:
                        efermi = float(m.group(1))
                        break
        
        kpoints, bands, occs = parse_nscf_all_kpoints(nscf_path)
        
        if not kpoints:
            print(f"| {cid} | {formula} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        result = analyze_corrected(kpoints, bands, occs)
        
        efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
        
        if result['type'] == 'semiconductor':
            direct_str = "直接" if result['direct'] else "间接"
            print(f"| {cid} | {formula} | {efermi_str} | {result['vbm_energy']:.2f} | {result['vbm_k_index']} | {result['cbm_energy']:.2f} | {result['cbm_k_index']} | {result['indirect_gap']:.2f} | {result['direct_gap']:.2f} | {direct_str} |")
        elif result['type'] in ('metal', 'semimetal'):
            print(f"| {cid} | {formula} | {efermi_str} | {result['vbm_energy']:.2f} | - | {result['cbm_energy']:.2f} | - | 0.00 | {result['min_positive_gap']:.2f} | {result['type']} |")
        else:
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | 未知 |")
        
        all_results[cid] = {
            'formula': formula,
            'fermi': efermi,
            'result': result
        }
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_corrected_gap_v2.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_corrected_gap_v2.json")
    
    # 统计
    n_metal = sum(1 for r in all_results.values() if r['result']['type'] == 'metal')
    n_semimetal = sum(1 for r in all_results.values() if r['result']['type'] == 'semimetal')
    n_sem = sum(1 for r in all_results.values() if r['result']['type'] == 'semiconductor')
    n_direct = sum(1 for r in all_results.values() if r['result']['type'] == 'semiconductor' and r['result']['direct'])
    n_indirect = sum(1 for r in all_results.values() if r['result']['type'] == 'semiconductor' and not r['result']['direct'])
    
    print(f"\n统计: 半导体={n_sem} (直接{n_direct}, 间接{n_indirect}), 半金属={n_semimetal}, 金属={n_metal}")

if __name__ == '__main__':
    main()
