import re, os

def check_partial_occ(path):
    with open(path, 'r') as f:
        text = f.read()
    
    pattern = r'k\s*=\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+\(\s*\d+\s+PW[s]?\)\s+bands \(ev\):'
    matches = list(re.finditer(pattern, text))
    
    partial_count = 0
    max_partial_occ = 0
    
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end]
        
        lines = block.split('\n')
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
            
            if in_occ:
                for v in stripped.split():
                    try:
                        o = float(v)
                        if 0.01 < o < 0.99:
                            partial_count += 1
                            max_partial_occ = max(max_partial_occ, o)
                    except:
                        pass
    
    return partial_count, max_partial_occ

base = 'C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs'

crystals = ['0742', '0789', '0391', '0631', '0912', '0217', '0672', '0927', '0626', '0632']

print("| 编号 | 部分占据态数 | 最大部分占据 | 是否金属? |")
print("|------|------------|------------|----------|")

for cid in crystals:
    nscf_path = f'{base}/crystal_{cid}_band/nscf/crystal_{cid}_decoded.nscf.out'
    count, max_occ = check_partial_occ(nscf_path)
    is_metal = "是" if count > 0 else "否"
    print(f"| {cid} | {count} | {max_occ:.4f} | {is_metal} |")
