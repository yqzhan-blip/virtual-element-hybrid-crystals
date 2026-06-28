import subprocess
import sys
import time
import os
import shutil
import numpy as np

QE_BIN = r"C:/Users/zhan/WorkBuddy/2026-06-15-19-55-32/qe-7.5/bin/pw.exe"
SSSP_DIR = r"C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs/pseudos/SSSP"
SHARED_PSEUDOS = r"C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs/crystal_0631_band/pseudos"
BASE_DIR = r"C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/dft_inputs"
BALANCED_CIFS = r"C:/Users/zhan/WorkBuddy/2026-05-28-task-12/paper_results/balanced_cifs"

# Use pymatgen for CIF parsing (handles non-orthogonal cells correctly)
try:
    from pymatgen.core import Structure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

QUEUE = [
    ("0672", "FeH8C3Br3N", {"Fe": ("Fe", 55.845, "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF"), "Br": ("Br", 79.904, "br_pbe_v1.4.uspp.F.UPF"), "C": ("C", 12.0107, "C.pbe-n-kjpaw_psl.1.0.0.UPF"), "N": ("N", 14.0067, "N.pbe-n-radius_5.UPF"), "H": ("H", 1.00794, "H.pbe-rrkjus_psl.1.0.0.UPF")}),
    ("0927", "CoH12C2N2Cl4", {"Co": ("Co", 58.933, "co_pbe_v1.2.uspp.F.UPF"), "Cl": ("Cl", 35.453, "cl_pbe_v1.4.uspp.F.UPF"), "C": ("C", 12.0107, "C.pbe-n-kjpaw_psl.1.0.0.UPF"), "N": ("N", 14.0067, "N.pbe-n-radius_5.UPF"), "H": ("H", 1.00794, "H.pbe-rrkjus_psl.1.0.0.UPF")}),
    ("0626", "VH11C2N3Br6", {"V": ("V", 50.942, "v_pbe_v1.4.uspp.F.UPF"), "Br": ("Br", 79.904, "br_pbe_v1.4.uspp.F.UPF"), "C": ("C", 12.0107, "C.pbe-n-kjpaw_psl.1.0.0.UPF"), "N": ("N", 14.0067, "N.pbe-n-radius_5.UPF"), "H": ("H", 1.00794, "H.pbe-rrkjus_psl.1.0.0.UPF")}),
    ("0632", "CrH11C2N3Br6", {"Cr": ("Cr", 51.996, "cr_pbe_v1.5.uspp.F.UPF"), "Br": ("Br", 79.904, "br_pbe_v1.4.uspp.F.UPF"), "C": ("C", 12.0107, "C.pbe-n-kjpaw_psl.1.0.0.UPF"), "N": ("N", 14.0067, "N.pbe-n-radius_5.UPF"), "H": ("H", 1.00794, "H.pbe-rrkjus_psl.1.0.0.UPF")}),
]

STATE_FILE = os.path.join(BASE_DIR, "..", "dft_queue_state.txt")

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def read_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return f.read().strip()
    return None

def write_state(state):
    with open(STATE_FILE, 'w') as f:
        f.write(state)

def check_job_done(output_path):
    if not os.path.exists(output_path):
        return False
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        return "JOB DONE" in f.read()

def get_last_line(output_path, n=20):
    if not os.path.exists(output_path):
        return ""
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    return ''.join(lines[-n:])

def extract_fermi(output_path):
    if not os.path.exists(output_path):
        return None
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "the Fermi energy is" in line.lower():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.lower() == "is" and i+1 < len(parts):
                        try:
                            return float(parts[i+1])
                        except:
                            pass
    return None

def extract_total_energy(output_path):
    if not os.path.exists(output_path):
        return None
    energy = None
    with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "total energy" in line and "=" in line:
                try:
                    e = float(line.split("=")[-1].strip().split()[0])
                    energy = e
                except:
                    pass
    return energy

def parse_cif(cif_path):
    atoms = []
    with open(cif_path, 'r') as f:
        lines = f.readlines()
    in_loop = False
    headers = []
    for line in lines:
        line = line.strip()
        if line.startswith("loop_"):
            in_loop = True
            headers = []
            continue
        if in_loop and line.startswith("_"):
            headers.append(line)
        elif in_loop and line and not line.startswith("_") and not line.startswith("#"):
            if "_atom_site_type_symbol" in headers:
                parts = line.split()
                if len(parts) >= 6:
                    elem = parts[0]
                    x, y, z = parts[3], parts[4], parts[5]
                    atoms.append((elem, x, y, z))
        elif in_loop and line.startswith("_") == False and line:
            in_loop = False
    cell = {}
    for line in lines:
        for key in ["_cell_length_a", "_cell_length_b", "_cell_length_c"]:
            if line.startswith(key):
                cell[key] = line.split()[-1]
    return atoms, cell

def parse_cif_pymatgen(cif_path):
    """Parse CIF using pymatgen, returning cell matrix + fractional positions."""
    if not HAS_PYMATGEN:
        return None, None, None
    struct = Structure.from_file(cif_path, primitive=False)
    # Lattice matrix in angstrom (rows are lattice vectors)
    lattice_matrix = struct.lattice.matrix
    # Species and fractional coordinates
    species = [str(sp) for sp in struct.species]
    frac_coords = struct.frac_coords
    return lattice_matrix, species, frac_coords

def compute_nbnd(species_list, ntyp_elements):
    """Estimate nbnd as 1.2x the number of valence electrons / 2."""
    # Rough valence electron count from common elements
    valence = {
        'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6,
        'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6,
        'Cl': 7, 'Ar': 8, 'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6,
        'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12, 'Ga': 3,
        'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8, 'Rb': 1, 'Sr': 2,
        'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Ru': 8, 'Rh': 9, 'Pd': 10,
        'Ag': 11, 'Cd': 12, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7,
        'Xe': 8, 'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Eu': 9, 'Gd': 10,
        'Pb': 4, 'Bi': 5,
    }
    nelec = sum(valence.get(sp, 4) for sp in species_list)  # default 4 valence
    nbnd = max(int(nelec * 1.2 / 2) + 4, 20)
    return nbnd

def write_scf_input(crystal_id, formula, elements, cif_path, out_path):
    if HAS_PYMATGEN:
        lattice_matrix, species, frac_coords = parse_cif_pymatgen(cif_path)
        if lattice_matrix is None:
            log(f"ERROR: pymatgen failed to parse CIF for {crystal_id}")
            return False
        nat = len(species)
        ntyp = len(elements)
        nbnd = compute_nbnd(species, elements)
    else:
        atoms, cell = parse_cif(cif_path)
        if not atoms:
            log(f"ERROR: Failed to parse CIF for {crystal_id}")
            return False
        species = [a[0] for a in atoms]
        nat = len(atoms)
        ntyp = len(elements)
        nbnd = compute_nbnd(species, elements)
        lattice_matrix = None
        frac_coords = None

    lines = []
    lines.append("&CONTROL")
    lines.append(f"    calculation = 'scf'")
    lines.append(f"    prefix = 'crystal_{crystal_id}_decoded'")
    lines.append("    outdir = './outdir'")
    lines.append("    pseudo_dir = '../pseudos'")
    lines.append("    tprnfor = .true.")
    lines.append("    tstress = .true.")
    lines.append("    verbosity = 'high'")
    lines.append("/")
    lines.append("")
    lines.append("&SYSTEM")
    lines.append("    ibrav = 0")
    lines.append(f"    nat = {nat}")
    lines.append(f"    ntyp = {ntyp}")
    lines.append("    ecutwfc = 50")
    lines.append("    ecutrho = 400")
    lines.append("    occupations = 'smearing'")
    lines.append("    smearing = 'gaussian'")
    lines.append("    degauss = 0.01")
    lines.append(f"    nbnd = {nbnd}")
    lines.append("/")
    lines.append("")
    lines.append("&ELECTRONS")
    lines.append("    conv_thr = 1.0d-6")
    lines.append("    mixing_beta = 0.4")
    lines.append("    diagonalization = 'cg'")
    lines.append("/")
    lines.append("")
    lines.append("ATOMIC_SPECIES")
    for elem, (sym, mass, pp) in elements.items():
        lines.append(f"    {sym}  {mass}   {pp}")
    lines.append("")
    lines.append("CELL_PARAMETERS angstrom")
    if lattice_matrix is not None:
        for row in lattice_matrix:
            lines.append(f"    {row[0]:20.12f}   {row[1]:20.12f}   {row[2]:20.12f}")
    else:
        # Fallback: diagonal only (will fail for non-orthogonal cells)
        for key in ["_cell_length_a", "_cell_length_b", "_cell_length_c"]:
            lines.append(f"    {cell.get(key, '5.0')}   0.00000000   0.00000000")
    lines.append("")
    lines.append("ATOMIC_POSITIONS crystal")
    if frac_coords is not None:
        for sp, fc in zip(species, frac_coords):
            lines.append(f"    {sp}  {fc[0]:.12f}  {fc[1]:.12f}  {fc[2]:.12f}")
    else:
        for elem, x, y, z in atoms:
            lines.append(f"    {elem}  {x}  {y}  {z}")
    lines.append("")
    lines.append("K_POINTS automatic")
    lines.append("    4 4 4 0 0 0")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"SCF input written: nat={nat}, ntyp={ntyp}, nbnd={nbnd}")
    return True

def copy_pseudos(elements, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for elem, (sym, mass, pp) in elements.items():
        src = os.path.join(SSSP_DIR, pp)
        if not os.path.exists(src):
            src = os.path.join(SHARED_PSEUDOS, pp)
        if not os.path.exists(src):
            log(f"ERROR: Pseudopotential not found: {pp}")
            return False
        dst = os.path.join(target_dir, pp)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    return True

def run_scf(crystal_id, scf_dir):
    scf_in = os.path.join(scf_dir, f"crystal_{crystal_id}_decoded.scf.in")
    scf_out = os.path.join(scf_dir, f"crystal_{crystal_id}_decoded.scf.out")
    if os.path.exists(scf_out) and check_job_done(scf_out):
        log(f"SCF for {crystal_id} already completed")
        return True
    if os.path.exists(scf_out):
        last_mod = os.path.getmtime(scf_out)
        if time.time() - last_mod < 600:
            log(f"SCF for {crystal_id} appears to be running (last mod {int(time.time()-last_mod)}s ago)")
            return "running"
    cmd = f'cd "{scf_dir}" && "{QE_BIN}" < crystal_{crystal_id}_decoded.scf.in > crystal_{crystal_id}_decoded.scf.out 2>&1'
    log(f"Starting SCF for {crystal_id}")
    subprocess.Popen(cmd, shell=True)
    return "started"

def run_nscf(crystal_id, nscf_dir, scf_dir):
    nscf_in = os.path.join(nscf_dir, f"crystal_{crystal_id}_decoded.nscf.in")
    nscf_out = os.path.join(nscf_dir, f"crystal_{crystal_id}_decoded.nscf.out")
    if os.path.exists(nscf_out) and check_job_done(nscf_out):
        log(f"NSCF for {crystal_id} already completed")
        return True
    if os.path.exists(nscf_out):
        last_mod = os.path.getmtime(nscf_out)
        if time.time() - last_mod < 600:
            log(f"NSCF for {crystal_id} appears to be running")
            return "running"
    # Generate NSCF input from SCF input with proper modifications
    scf_in = os.path.join(scf_dir, f"crystal_{crystal_id}_decoded.scf.in")
    if not os.path.exists(scf_in):
        log(f"ERROR: SCF input not found for {crystal_id}")
        return False
    with open(scf_in, 'r') as f:
        scf_content = f.read()
    nscf_content = scf_content.replace("calculation = 'scf'", "calculation = 'nscf'")
    nscf_content = nscf_content.replace("outdir = './outdir'", "outdir = '../scf/outdir'")
    nscf_content = nscf_content.replace("conv_thr = 1.0d-6", "conv_thr = 1.0d-7")
    nscf_content = nscf_content.replace("diagonalization = 'cg'", "diagonalization = 'david'")
    nscf_content = nscf_content.replace("tprnfor = .true.", "tprnfor = .false.")
    nscf_content = nscf_content.replace("tstress = .true.", "tstress = .false.")
    nscf_content = nscf_content.replace("4 4 4 0 0 0", "6 6 6 0 0 0")
    with open(nscf_in, 'w') as f:
        f.write(nscf_content)
    cmd = f'cd "{nscf_dir}" && "{QE_BIN}" < crystal_{crystal_id}_decoded.nscf.in > crystal_{crystal_id}_decoded.nscf.out 2>&1'
    log(f"Starting NSCF for {crystal_id}")
    subprocess.Popen(cmd, shell=True)
    return "started"

def report_results(crystal_id, scf_out, nscf_out):
    log(f"=== RESULTS FOR crystal_{crystal_id} ===")
    energy = extract_total_energy(scf_out)
    fermi = extract_fermi(scf_out)
    log(f"Total Energy: {energy} Ry")
    log(f"Fermi Energy: {fermi} eV")
    if os.path.exists(nscf_out):
        with open(nscf_out, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if "k = 0.0000 0.0000 0.0000" in content:
            idx = content.find("k = 0.0000 0.0000 0.0000")
            block = content[idx:idx+2000]
            lines = block.split('\n')
            bands = []
            occs = []
            for i, line in enumerate(lines):
                if "bands (ev):" in line and i+1 < len(lines):
                    band_line = lines[i+1].strip()
                    bands.extend([float(x) for x in band_line.split() if x])
                if "occupation numbers" in line and i+1 < len(lines):
                    occ_line = lines[i+1].strip()
                    occs.extend([float(x) for x in occ_line.split() if x])
            if bands and occs and len(bands) == len(occs):
                homo = None
                lumo = None
                for b, o in zip(bands, occs):
                    if o > 0.5:
                        homo = b
                    elif lumo is None and o < 0.5:
                        lumo = b
                if homo and lumo and fermi:
                    gap = lumo - homo if lumo > homo else 0.0
                    if homo < fermi < lumo:
                        log(f"Band Gap: {gap:.2f} eV (HOMO={homo:.3f}, LUMO={lumo:.3f})")
                        log(f"Property: {'Semiconductor' if gap > 0.1 else 'Metallic'}")
                    else:
                        log(f"Fermi level not in gap - Metallic (HOMO={homo:.3f}, LUMO={lumo:.3f})")
                        log(f"Property: Metallic")
    log(f"=== END RESULTS ===")

def main():
    log("DFT Calculator Manager started")
    current = read_state()
    if not current:
        current = QUEUE[0][0]
        write_state(current)
        log(f"No active state, starting with crystal_{current}")
    log(f"Current crystal: {current}")
    queue_idx = None
    for i, (cid, _, _) in enumerate(QUEUE):
        if cid == current:
            queue_idx = i
            break
    if queue_idx is None:
        log(f"Current crystal {current} not in queue, checking if completed")
        scf_out = os.path.join(BASE_DIR, f"crystal_{current}_band", "scf", f"crystal_{current}_decoded.scf.out")
        nscf_out = os.path.join(BASE_DIR, f"crystal_{current}_band", "nscf", f"crystal_{current}_decoded.nscf.out")
        if check_job_done(scf_out) and check_job_done(nscf_out):
            for i, (cid, _, _) in enumerate(QUEUE):
                if cid == current:
                    queue_idx = i
                    break
            if queue_idx is not None and queue_idx + 1 < len(QUEUE):
                current = QUEUE[queue_idx + 1][0]
                write_state(current)
                log(f"Moving to next crystal: {current}")
            else:
                log("All calculations in queue complete!")
                return
        else:
            log(f"ERROR: Cannot find crystal {current} in queue")
            return
    crystal_id, formula, elements = QUEUE[queue_idx]
    work_dir = os.path.join(BASE_DIR, f"crystal_{crystal_id}_band")
    scf_dir = os.path.join(work_dir, "scf")
    nscf_dir = os.path.join(work_dir, "nscf")
    pseudo_dir = os.path.join(work_dir, "pseudos")
    scf_out = os.path.join(scf_dir, f"crystal_{crystal_id}_decoded.scf.out")
    nscf_out = os.path.join(nscf_dir, f"crystal_{crystal_id}_decoded.nscf.out")
    if check_job_done(scf_out):
        log(f"SCF for {crystal_id} COMPLETE")
        if check_job_done(nscf_out):
            log(f"NSCF for {crystal_id} COMPLETE")
            report_results(crystal_id, scf_out, nscf_out)
            if queue_idx + 1 < len(QUEUE):
                next_id = QUEUE[queue_idx + 1][0]
                write_state(next_id)
                log(f"Moving to next crystal: {next_id}")
                next_cid, next_formula, next_elements = QUEUE[queue_idx + 1]
                next_work = os.path.join(BASE_DIR, f"crystal_{next_cid}_band")
                next_scf = os.path.join(next_work, "scf")
                next_nscf = os.path.join(next_work, "nscf")
                next_pseudo = os.path.join(next_work, "pseudos")
                next_cif = os.path.join(BALANCED_CIFS, f"crystal_{next_cid}_decoded.cif")
                os.makedirs(next_scf, exist_ok=True)
                os.makedirs(next_nscf, exist_ok=True)
                os.makedirs(next_pseudo, exist_ok=True)
                os.makedirs(os.path.join(next_scf, "outdir"), exist_ok=True)
                if not copy_pseudos(next_elements, next_pseudo):
                    log(f"ERROR: Failed to copy pseudos for {next_cid}")
                    return
                scf_in = os.path.join(next_scf, f"crystal_{next_cid}_decoded.scf.in")
                if not os.path.exists(scf_in):
                    if not write_scf_input(next_cid, next_formula, next_elements, next_cif, scf_in):
                        log(f"ERROR: Failed to write SCF input for {next_cid}")
                        return
                result = run_scf(next_cid, next_scf)
                log(f"Started SCF for {next_cid}: {result}")
            else:
                log("ALL CALCULATIONS COMPLETE!")
        else:
            log(f"SCF done, starting NSCF for {crystal_id}")
            result = run_nscf(crystal_id, nscf_dir, scf_dir)
            log(f"NSCF result: {result}")
    else:
        if os.path.exists(scf_out):
            last_mod = os.path.getmtime(scf_out)
            age = time.time() - last_mod
            if age < 600:
                log(f"SCF for {crystal_id} is running (last output {int(age)}s ago)")
            else:
                log(f"SCF for {crystal_id} STALLED (last output {int(age)}s ago)")
                log(f"Last lines: {get_last_line(scf_out)}")
                result = run_scf(crystal_id, scf_dir)
                log(f"Restart attempt: {result}")
        else:
            log(f"SCF for {crystal_id} not started yet")
            os.makedirs(scf_dir, exist_ok=True)
            os.makedirs(nscf_dir, exist_ok=True)
            os.makedirs(pseudo_dir, exist_ok=True)
            os.makedirs(os.path.join(scf_dir, "outdir"), exist_ok=True)
            cif_path = os.path.join(BALANCED_CIFS, f"crystal_{crystal_id}_decoded.cif")
            if not copy_pseudos(elements, pseudo_dir):
                log(f"ERROR: Failed to copy pseudos")
                return
            scf_in = os.path.join(scf_dir, f"crystal_{crystal_id}_decoded.scf.in")
            if not os.path.exists(scf_in):
                if not write_scf_input(crystal_id, formula, elements, cif_path, scf_in):
                    log(f"ERROR: Failed to write SCF input")
                    return
            result = run_scf(crystal_id, scf_dir)
            log(f"Started SCF: {result}")

if __name__ == "__main__":
    main()
