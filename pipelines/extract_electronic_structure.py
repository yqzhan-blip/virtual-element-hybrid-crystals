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

def analyze_electronic_structure(kpoints, bands, occs):
    """分析电子结构：判断金属/半导体，计算带隙"""
    if not kpoints:
        return None
    
    nk = len(kpoints)
    
    # 1. 检查是否有部分占据态 (0.01 < occ < 0.99)
    partial_occ = False
    partial_occ_details = []
    for ik in range(nk):
        for ib, (e, o) in enumerate(zip(bands[ik], occs[ik])):
            if 0.01 < o < 0.99:
                partial_occ = True
                partial_occ_details.append((ik, ib, e, o))
    
    if partial_occ:
        # 金属 - 找到穿越Fermi面的能带
        # 全局最高占据态 (occ >= 0.5) = "HOMO"
        # 全局最低空态 (occ < 0.5) = "LUMO"
        # 但金属的"带隙"是伪带隙，通常取最小直接gap或标记为0
        
        all_occ = []
        all_empty = []
        for ik in range(nk):
            for e, o in zip(bands[ik], occs[ik]):
                if o >= 0.5:
                    all_occ.append((e, ik))
                else:
                    all_empty.append((e, ik))
        
        if all_occ and all_empty:
            homo = max(all_occ, key=lambda x: x[0])
            lumo = min(all_empty, key=lambda x: x[0])
            pseudo_gap = lumo[0] - homo[0]
            
            return {
                'type': 'metal',
                'partial_occ_count': len(partial_occ_details),
                'partial_occ_examples': partial_occ_details[:5],
                'homo_energy': homo[0],
                'homo_k': kpoints[homo[1]],
                'lumo_energy': lumo[0],
                'lumo_k': kpoints[lumo[1]],
                'pseudo_gap': pseudo_gap,  # 可能是负值
                'direct_gap': 0.0,  # 金属的直接带隙定义为0
            }
    
    # 2. 半导体/绝缘体 - 清晰的分割 (occ=1.0 vs occ=0.0)
    direct_gaps = []
    
    for ik in range(nk):
        if len(bands[ik]) != len(occs[ik]):
            continue
        
        # HOMO: 最高能量的占据态 (occ >= 0.99)
        homo_candidates = [(e, occ) for e, occ in zip(bands[ik], occs[ik]) if occ >= 0.99]
        # LUMO: 最低能量的空态 (occ <= 0.01)
        lumo_candidates = [(e, occ) for e, occ in zip(bands[ik], occs[ik]) if occ <= 0.01]
        
        if not homo_candidates or not lumo_candidates:
            continue
        
        homo = max(homo_candidates, key=lambda x: x[0])
        lumo = min(lumo_candidates, key=lambda x: x[0])
        
        gap = lumo[0] - homo[0]
        if gap > 0:
            direct_gaps.append((ik, gap, homo[0], lumo[0]))
    
    if not direct_gaps:
        return {'type': 'unknown'}
    
    # 最小直接带隙
    min_gap_info = min(direct_gaps, key=lambda x: x[1])
    min_ik, min_gap, min_homo, min_lumo = min_gap_info
    
    # 间接带隙：全局最高HOMO和全局最低LUMO
    all_homo = [(h, ik) for _, _, h, _ in direct_gaps]
    all_lumo = [(l, ik) for _, _, _, l in direct_gaps]
    
    global_homo = max(all_homo, key=lambda x: x[0])
    global_lumo = min(all_lumo, key=lambda x: x[0])
    
    indirect_gap = global_lumo[0] - global_homo[0]
    
    # 是否直接带隙：全局HOMO和全局LUMO是否在同一k点
    direct = (global_homo[1] == global_lumo[1])
    
    return {
        'type': 'semiconductor',
        'gap': indirect_gap,  # 间接带隙
        'direct_gap': min_gap,  # 最小直接带隙
        'direct': direct,
        'homo_energy': global_homo[0],  # VBM
        'homo_k': kpoints[global_homo[1]],
        'homo_k_index': global_homo[1],
        'lumo_energy': global_lumo[0],  # CBM
        'lumo_k': kpoints[global_lumo[1]],
        'lumo_k_index': global_lumo[1],
        'min_direct_k': kpoints[min_ik],
        'min_direct_k_index': min_ik,
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
    
    print("| 编号 | 成分 | Fermi(eV) | 类型 | HOMO/VBM(eV) | LUMO/CBM(eV) | 间接带隙(eV) | 直接带隙? | 最小直接带隙(eV) | 部分占据数 |")
    print("|------|------|-----------|------|-------------|-------------|-------------|----------|----------------|----------|")
    
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
        
        result = analyze_electronic_structure(kpoints, bands, occs)
        
        efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
        
        if result['type'] == 'metal':
            partial_count = result['partial_occ_count']
            pseudo_gap = result['pseudo_gap']
            print(f"| {cid} | {formula} | {efermi_str} | **金属** | {result['homo_energy']:.2f} | {result['lumo_energy']:.2f} | {pseudo_gap:.2f} | - | 0.00 | {partial_count} |")
        elif result['type'] == 'semiconductor':
            direct_str = "是" if result['direct'] else "否"
            print(f"| {cid} | {formula} | {efermi_str} | 半导体 | {result['homo_energy']:.2f} | {result['lumo_energy']:.2f} | {result['gap']:.2f} | {direct_str} | {result['direct_gap']:.2f} | 0 |")
        else:
            print(f"| {cid} | {formula} | {efermi_str} | 未知 | N/A | N/A | N/A | N/A | N/A | N/A |")
        
        all_results[cid] = {
            'formula': formula,
            'fermi': efermi,
            'result': result
        }
    
    # 统计
    n_metal = sum(1 for r in all_results.values() if r['result']['type'] == 'metal')
    n_sem = sum(1 for r in all_results.values() if r['result']['type'] == 'semiconductor')
    
    print(f"\n统计: 金属={n_metal}, 半导体={n_sem}")
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_electronic_structure.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_electronic_structure.json")

if __name__ == '__main__':
    main()
