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
    """使用HOMO/LUMO方法分析带隙 (occ > 0.5为占据, occ < 0.5为空)"""
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
        if gap > 0:
            direct_gaps.append((ik, gap, homo[0], lumo[0], homo[1], lumo[1]))
    
    if not direct_gaps:
        return None
    
    # 最小直接带隙
    min_gap_info = min(direct_gaps, key=lambda x: x[1])
    min_ik, min_gap, min_homo, min_lumo, min_homo_occ, min_lumo_occ = min_gap_info
    
    # 全局VBM和CBM
    all_homo = [(h, ik) for _, _, h, _, _, _ in direct_gaps]
    all_lumo = [(l, ik) for _, _, _, l, _, _ in direct_gaps]
    
    global_vbm = max(all_homo, key=lambda x: x[0])
    global_cbm = min(all_lumo, key=lambda x: x[0])
    
    indirect_gap = global_cbm[0] - global_vbm[0]
    
    # 是否直接带隙：全局VBM和全局CBM是否在同一k点
    direct = (global_vbm[1] == global_cbm[1])
    
    # 最小直接带隙值
    min_direct_gap = min_gap
    
    return {
        'gap': indirect_gap,  # 间接带隙
        'direct_gap': min_direct_gap,  # 最小直接带隙
        'direct': direct,  # 是否直接带隙
        'vbm_energy': global_vbm[0],
        'vbm_k': kpoints[global_vbm[1]],
        'vbm_k_index': global_vbm[1],
        'cbm_energy': global_cbm[0],
        'cbm_k': kpoints[global_cbm[1]],
        'cbm_k_index': global_cbm[1],
        'min_direct_k': kpoints[min_ik],
        'min_direct_k_index': min_ik,
        'min_direct_homo': min_homo,
        'min_direct_lumo': min_lumo,
        'min_direct_homo_occ': min_homo_occ,
        'min_direct_lumo_occ': min_lumo_occ,
    }

def generate_dos(kpoints, bands, occs, efermi, de=0.05, emin=-10, emax=10):
    """生成简单的DOS数据"""
    all_energies = []
    for ik in range(len(kpoints)):
        all_energies.extend(bands[ik])
    
    all_energies = np.array(all_energies)
    bins = np.arange(emin, emax + de, de)
    dos, _ = np.histogram(all_energies, bins=bins)
    
    occ_energies = []
    unocc_energies = []
    for ik in range(len(kpoints)):
        for e, o in zip(bands[ik], occs[ik]):
            if o > 0.5:
                occ_energies.append(e)
            elif o < 0.5:
                unocc_energies.append(e)
    
    return bins[:-1], dos, occ_energies, unocc_energies

def plot_results(bands_data, gap_info, efermi, output_dir, prefix):
    """生成DOS图和能带分布图"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return False
    
    kpoints, bands, occs = bands_data
    
    # 1. DOS图
    bins, dos, occ_e, unocc_e = generate_dos(kpoints, bands, occs, efermi)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(bins, 0, dos, alpha=0.5, color='blue', label='Total DOS')
    ax.axvline(efermi, color='red', lw=1, ls='--', label=f'E_F = {efermi:.2f} eV')
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('DOS (arb. units)', fontsize=12)
    ax.set_title(f'{prefix} Density of States', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    dos_path = os.path.join(output_dir, f'{prefix}.dos.svg')
    fig.savefig(dos_path)
    print(f"  Saved DOS plot: {dos_path}")
    plt.close()
    
    # 2. 能带分布图 (简化：显示所有k点的能量分布)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for ik in range(len(kpoints)):
        for e, o in zip(bands[ik], occs[ik]):
            color = 'blue' if o > 0.5 else 'red' if o < 0.5 else 'gray'
            ax.plot(ik, e, 'o', color=color, markersize=1.5, alpha=0.6)
    
    ax.axhline(efermi, color='green', lw=1.5, ls='--', label='E_F')
    if gap_info:
        ax.axhline(gap_info['vbm_energy'], color='blue', lw=1, ls='-', alpha=0.5, label='VBM')
        ax.axhline(gap_info['cbm_energy'], color='red', lw=1, ls='-', alpha=0.5, label='CBM')
    
    ax.set_xlabel('k-point index', fontsize=12)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title(f'{prefix} Band Distribution (all k-points)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    band_path = os.path.join(output_dir, f'{prefix}.bands_all.svg')
    fig.savefig(band_path)
    print(f"  Saved band distribution plot: {band_path}")
    plt.close()
    
    return True

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
    
    print("| 编号 | 成分 | Fermi(eV) | VBM(eV) | CBM(eV) | 间接带隙(eV) | 直接带隙? | 最小直接带隙(eV) | VBM_k | CBM_k |")
    print("|------|------|-----------|---------|---------|-------------|----------|----------------|-------|-------|")
    
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
            vbm_k_str = f"({gap_info['vbm_k'][0]:.2f},{gap_info['vbm_k'][1]:.2f},{gap_info['vbm_k'][2]:.2f})"
            cbm_k_str = f"({gap_info['cbm_k'][0]:.2f},{gap_info['cbm_k'][1]:.2f},{gap_info['cbm_k'][2]:.2f})"
            
            print(f"| {cid} | {formula} | {efermi_str} | {gap_info['vbm_energy']:.2f} | {gap_info['cbm_energy']:.2f} | {gap_info['gap']:.2f} | {direct_str} | {gap_info['direct_gap']:.2f} | {vbm_k_str} | {cbm_k_str} |")
            
            all_results[cid] = {
                'formula': formula,
                'fermi': efermi,
                'vbm': gap_info['vbm_energy'],
                'vbm_k': gap_info['vbm_k'],
                'vbm_k_index': gap_info['vbm_k_index'],
                'cbm': gap_info['cbm_energy'],
                'cbm_k': gap_info['cbm_k'],
                'cbm_k_index': gap_info['cbm_k_index'],
                'gap': gap_info['gap'],
                'direct': gap_info['direct'],
                'direct_gap': gap_info['direct_gap'],
                'min_direct_k': gap_info['min_direct_k'],
                'min_direct_k_index': gap_info['min_direct_k_index'],
            }
            
            # 生成图
            results_dir = f'{base}/crystal_{cid}_band/results'
            os.makedirs(results_dir, exist_ok=True)
            plot_results((kpoints, bands, occs), gap_info, efermi, results_dir, f'crystal_{cid}_decoded')
        else:
            efermi_str = f"{efermi:.2f}" if efermi is not None else "N/A"
            print(f"| {cid} | {formula} | {efermi_str} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_gap_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n结果已保存到: dft_gap_analysis.json")

if __name__ == '__main__':
    main()
