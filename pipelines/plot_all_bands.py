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

def analyze_band_structure(kpoints, bands, occs):
    """分析能带结构：判断金属/半导体，计算带隙"""
    if not kpoints:
        return None
    
    nk = len(kpoints)
    local_gaps = []  # (k_index, gap, vbm, cbm)
    
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
    
    has_negative_gap = any(gap < 0 for _, gap, _, _ in local_gaps)
    
    if has_negative_gap:
        positive_gaps = [(ik, gap, vbm, cbm) for ik, gap, vbm, cbm in local_gaps if gap > 0]
        
        if positive_gaps:
            min_positive = min(positive_gaps, key=lambda x: x[1])
            return {
                'type': 'metal' if min_positive[1] < 0.1 else 'semimetal',
                'direct_gap': 0.0,
                'indirect_gap': 0.0,
                'min_positive_gap': min_positive[1],
                'min_positive_k': kpoints[min_positive[0]],
                'min_positive_k_index': min_positive[0],
                'vbm_energy': max(vbm for _, _, vbm, _ in local_gaps),
                'cbm_energy': min(cbm for _, _, _, cbm in local_gaps),
                'has_negative_gap': True,
            }
        else:
            return {'type': 'metal', 'direct_gap': 0.0, 'indirect_gap': 0.0, 'has_negative_gap': True}
    
    min_direct = min(local_gaps, key=lambda x: x[1])
    
    all_vbm = [(vbm, ik) for ik, _, vbm, _ in local_gaps]
    all_cbm = [(cbm, ik) for ik, _, _, cbm in local_gaps]
    
    global_vbm = max(all_vbm, key=lambda x: x[0])
    global_cbm = min(all_cbm, key=lambda x: x[0])
    
    indirect_gap = global_cbm[0] - global_vbm[0]
    direct_gap = min_direct[1]
    direct = (global_vbm[1] == global_cbm[1])
    
    return {
        'type': 'semiconductor',
        'direct_gap': direct_gap,
        'indirect_gap': indirect_gap,
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

def plot_band_distribution(kpoints, bands, occs, result, efermi, output_path, title):
    """生成能带分布图：所有k点的能量散点图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    nk = len(kpoints)
    
    # 收集所有能级数据用于绘图
    all_occ_energies = []
    all_empty_energies = []
    all_partial_energies = []
    
    for ik in range(nk):
        for e, o in zip(bands[ik], occs[ik]):
            if o >= 0.99:
                all_occ_energies.append((ik, e))
            elif o <= 0.01:
                all_empty_energies.append((ik, e))
            else:
                all_partial_energies.append((ik, e))
    
    # 绘制散点
    if all_occ_energies:
        x, y = zip(*all_occ_energies)
        ax.scatter(x, y, c='blue', s=3, alpha=0.6, label='Occupied', rasterized=True)
    
    if all_empty_energies:
        x, y = zip(*all_empty_energies)
        ax.scatter(x, y, c='red', s=3, alpha=0.6, label='Empty', rasterized=True)
    
    if all_partial_energies:
        x, y = zip(*all_partial_energies)
        ax.scatter(x, y, c='gray', s=3, alpha=0.8, label='Partial occupation', rasterized=True)
    
    # 绘制VBM/CBM线
    if result['type'] == 'semiconductor':
        ax.axhline(result['vbm_energy'], color='blue', lw=1.5, ls='--', alpha=0.7, label=f'VBM={result["vbm_energy"]:.2f} eV')
        ax.axhline(result['cbm_energy'], color='red', lw=1.5, ls='--', alpha=0.7, label=f'CBM={result["cbm_energy"]:.2f} eV')
    
    # 绘制Fermi能级
    if efermi is not None:
        ax.axhline(efermi, color='green', lw=2, ls='-.', alpha=0.8, label=f'E_F={efermi:.2f} eV')
    
    ax.set_xlabel('k-point index', fontsize=14)
    ax.set_ylabel('Energy (eV)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.2)
    
    # 设置合适的Y轴范围（集中在Fermi能级附近）
    if efermi is not None:
        ax.set_ylim([efermi - 5, efermi + 5])
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True

def plot_dos(kpoints, bands, occs, output_path, title, efermi=None):
    """生成DOS图"""
    all_energies = []
    for ik in range(len(kpoints)):
        all_energies.extend(bands[ik])
    
    if not all_energies:
        return False
    
    all_energies = np.array(all_energies)
    
    emin = np.min(all_energies) - 2
    emax = np.max(all_energies) + 2
    de = 0.05
    bins = np.arange(emin, emax + de, de)
    
    dos, edges = np.histogram(all_energies, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(centers, 0, dos, alpha=0.5, color='steelblue', label='Total DOS')
    
    if efermi is not None:
        ax.axvline(efermi, color='red', lw=1.5, ls='--', label=f'E_F={efermi:.2f} eV')
    
    ax.set_xlabel('Energy (eV)', fontsize=14)
    ax.set_ylabel('DOS (arb. units)', fontsize=14)
    ax.set_title(f'{title} DOS', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True

def main():
    base = 'C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs'
    out_dir = 'C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/figures'
    os.makedirs(out_dir, exist_ok=True)
    
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
    
    print("| 编号 | 成分 | 类型 | 直接带隙(eV) | 间接带隙(eV) | 图文件 |")
    print("|------|------|------|-------------|-------------|--------|")
    
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
            print(f"| {cid} | {formula} | 解析失败 | N/A | N/A | - |")
            continue
        
        result = analyze_band_structure(kpoints, bands, occs)
        
        if result['type'] == 'semiconductor':
            type_str = "半导体"
            gap_str = f"{result['direct_gap']:.2f}"
            indirect_str = f"{result['indirect_gap']:.2f}"
        elif result['type'] == 'metal':
            type_str = "金属"
            gap_str = "0.00"
            indirect_str = "0.00"
        elif result['type'] == 'semimetal':
            type_str = "半金属"
            gap_str = f"{result['min_positive_gap']:.2f}"
            indirect_str = "0.00"
        else:
            type_str = "未知"
            gap_str = "N/A"
            indirect_str = "N/A"
        
        # 生成图
        band_plot = f'{out_dir}/crystal_{cid}_bands.svg'
        dos_plot = f'{out_dir}/crystal_{cid}_dos.svg'
        
        plot_band_distribution(kpoints, bands, occs, result, efermi, band_plot, f'crystal_{cid} ({formula}) Band Distribution')
        plot_dos(kpoints, bands, occs, dos_plot, f'crystal_{cid} ({formula})', efermi)
        
        print(f"| {cid} | {formula} | {type_str} | {gap_str} | {indirect_str} | bands.svg, dos.svg |")
        
        all_results[cid] = {
            'formula': formula,
            'fermi': efermi,
            'result': result
        }
    
    # 保存JSON
    with open('C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_band_plots.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n所有图已保存到: {out_dir}/")

if __name__ == '__main__':
    main()
