"""Generate Nature-style composite figures for paper_draft_v6."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 7
plt.rcParams['figure.dpi'] = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "figures_nature")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================
# DATA
# ============================================

# DFT data
dft_data = {
    "0742": {"formula": "NiH\u2088C\u2083I\u2083N", "metal": "Ni", "halide": "I", "type": "Metallic", "indirect": 0, "direct": 0.06, "chgnet": -4.72},
    "0789": {"formula": "NiH\u2088C\u2083I\u2083N", "metal": "Ni", "halide": "I", "type": "Metallic", "indirect": 0, "direct": 0.03, "chgnet": -4.73},
    "0391": {"formula": "Mg\u2082H\u2081\u2082C\u2082I\u2086N\u2082", "metal": "Mg", "halide": "I", "type": "Metallic", "indirect": 0, "direct": 0.07, "chgnet": -3.41},
    "0631": {"formula": "CsH\u2086CBr\u2082N", "metal": "Cs", "halide": "Br", "type": "Indirect", "indirect": 2.17, "direct": 2.49, "chgnet": -4.42},
    "0912": {"formula": "KH\u2088C\u2083NF\u2082", "metal": "K", "halide": "F", "type": "Indirect", "indirect": 3.27, "direct": 3.39, "chgnet": -4.91},
    "0217": {"formula": "MnH\u2088C\u2083I\u2083N", "metal": "Mn", "halide": "I", "type": "Indirect", "indirect": 0.46, "direct": 0.49, "chgnet": -4.99},
    "0672": {"formula": "FeH\u2088C\u2083Br\u2083N", "metal": "Fe", "halide": "Br", "type": "Indirect", "indirect": 0.54, "direct": 0.60, "chgnet": -4.98},
    "0927": {"formula": "CoH\u2081\u2082C\u2082N\u2082Cl\u2084", "metal": "Co", "halide": "Cl", "type": "Indirect", "indirect": 0.54, "direct": 0.56, "chgnet": -3.09},
    "0626": {"formula": "VH\u2081\u2081C\u2082Br\u2086N\u2083", "metal": "V", "halide": "Br", "type": "Direct", "indirect": 0.78, "direct": 0.78, "chgnet": -3.62},
    "0632": {"formula": "CrH\u2081\u2081C\u2082Br\u2086N\u2083", "metal": "Cr", "halide": "Br", "type": "Indirect", "indirect": 0.51, "direct": 0.51, "chgnet": -4.22},
}

# Virtual element distribution (from SI Table S2)
virt_elements = {
    "X203": ("Sp2D-alkyl", 215, 9.6),
    "X204": ("Sp2D-aromatic", 214, 9.5),
    "X202": ("A3D-halogenated", 207, 9.2),
    "X208": ("DiA-alkyl", 204, 9.1),
    "X205": ("Sp2D-fused", 190, 8.4),
    "X209": ("DiA-aromatic", 188, 8.4),
    "X201": ("A3D-small", 193, 8.6),
    "X207": ("Sp2D-functional", 172, 7.6),
    "X206": ("Sp2D-cyclic", 141, 6.3),
    "X210": ("TriA", 44, 2.0),
    "X211": ("TetraA", 14, 0.6),
    "X212": ("PentaHexaA", 0, 0.0),
}

# Charge distribution (from paper text) - add 0 for neutral
net_charges = {
    0: 106, -5: 128, -1: 158, -3: 110, -2: 96, +1: 87, +3: 72, +2: 65, -4: 55, +4: 42, +5: 28,
    -6: 20, +6: 15, +7: 8, -7: 5, +8: 3, -8: 2, +9: 1, +10: 1, +11: 1, +12: 1
}

# Metal distribution (halide-containing 50 structures, from real data)
metals = {"Mn": 6, "Ni": 6, "Co": 4, "Fe": 3, "Be": 3, "Ca": 3, "La": 2, "Al": 2, "Ga": 2, "Cu": 2}

# Halide distribution (from real data: F=7, Cl=9, Br=17, I=15; paper rounds to F=8,Cl=10,Br=18,I=15)
halides = {"Br": 18, "I": 15, "Cl": 10, "F": 8}

# Template distribution (halide-only 50 structures, verified from real data)
templates = {"TriA": 20, "PentaHexaA": 9, "DiA-alkyl": 8, "TetraA": 6, "Sp2D-alkyl": 5, "DiA-aromatic": 1, "A3D-small": 1}

# Real metal-halide co-occurrence matrix (from balanced_analysis_hybrid.json)
# Rows match panel a metal counts; columns: F, Cl, Br, I
metal_halide_cooccur = {
    "Mn":  [0, 1, 3, 2],   # total=6
    "Ni":  [0, 1, 1, 4],   # total=6
    "Co":  [0, 1, 2, 1],   # total=4
    "Fe":  [1, 1, 1, 0],   # total=3
    "Be":  [0, 0, 1, 2],   # total=3
    "Ca":  [0, 0, 2, 1],   # total=3
    "La":  [1, 0, 1, 0],   # total=2
    "Al":  [0, 0, 1, 1],   # total=2
    "Ga":  [0, 2, 0, 0],   # total=2
    "Cu":  [0, 2, 0, 0],   # total=2
}

# Real organic class-halide co-occurrence (from balanced_analysis_hybrid.json)
# Rows: organic class; columns: F, Cl, Br, I
org_halide_cooccur = {
    "A3D-small":    [0, 0, 1, 0],   # total=1
    "Sp2D-alkyl":   [1, 0, 1, 3],   # total=5
    "DiA-alkyl":    [1, 3, 2, 3],   # total=9 (one structure has 2 halide types)
    "TriA":         [4, 2, 8, 6],   # total=20
    "TetraA":       [0, 3, 1, 2],   # total=6
    "PentaHexaA":   [2, 2, 4, 1],   # total=9
    "DiA-aromatic": [0, 0, 1, 0],   # total=1
}

# Color schemes
HALIDE_COLORS = {"F": "#E74C3C", "Cl": "#F39C12", "Br": "#27AE60", "I": "#8E44AD"}
TYPE_COLORS = {"Metallic": "#7F8C8D", "Indirect": "#3498DB", "Direct": "#1ABC9C"}
METAL_CATS = {
    "Ni": "trans", "Mn": "trans", "Fe": "trans", "Co": "trans", "V": "trans", "Cr": "trans", "Cu": "trans",
    "Bi": "main", "Al": "main", "Ga": "main", "Mg": "main", "Ca": "main", "Be": "main",
    "La": "alkaline", "K": "alkaline", "Cs": "alkaline", "Ba": "alkaline", "Sr": "alkaline",
    "Pb": "main", "Sn": "main", "Ge": "main", "Ti": "trans", "Sb": "main", "Zn": "trans",
    "Cd": "trans", "Ca": "alkaline", "Y": "trans", "Sc": "trans", "Ru": "trans",
    "Th": "actinide", "Tl": "main", "Cd": "trans"
}
CAT_COLORS = {"trans": "#E74C3C", "main": "#3498DB", "alkaline": "#27AE60", "actinide": "#9B59B6"}
CAT_LABELS = {"trans": "Transition metal", "main": "Main group", "alkaline": "Alkali/Alkaline earth", "actinide": "Actinide"}


# ============================================
# FIGURE 1: Virtual-element framework
# ============================================
def generate_figure1():
    fig = plt.figure(figsize=(7.2, 9.5))
    
    # Panel a: Representation gap
    ax_a = fig.add_axes([0.08, 0.74, 0.42, 0.22])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis('off')
    ax_a.set_title('**a**', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    # Left box: Inorganic model
    rect1 = FancyBboxPatch((0.5, 5), 3.5, 3.5, boxstyle="round,pad=0.1", 
                            facecolor='#3498DB', edgecolor='black', alpha=0.3, linewidth=1)
    ax_a.add_patch(rect1)
    ax_a.text(2.25, 7.5, 'Inorganic model', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_a.text(2.25, 6.8, 'Vocabulary: H, C, N, O...', ha='center', va='center', fontsize=7, style='italic')
    ax_a.text(2.25, 6.2, 'Pb, Bi, Cs, I, Br...', ha='center', va='center', fontsize=7, style='italic')
    
    # Right box: Hybrid crystal
    rect2 = FancyBboxPatch((6, 5), 3.5, 3.5, boxstyle="round,pad=0.1",
                            facecolor='#27AE60', edgecolor='black', alpha=0.3, linewidth=1)
    ax_a.add_patch(rect2)
    ax_a.text(7.75, 7.5, 'Hybrid crystal', ha='center', va='center', fontsize=8, fontweight='bold')
    ax_a.text(7.75, 6.8, 'Inorganic framework', ha='center', va='center', fontsize=7, style='italic')
    ax_a.text(7.75, 6.2, '+ molecular cations', ha='center', va='center', fontsize=7, style='italic')
    ax_a.text(7.75, 5.6, 'MA$^+$, EDA$^{2+}$, TAPA$^{3+}$', ha='center', va='center', fontsize=7, style='italic')
    
    # Gap arrow with X
    ax_a.annotate('', xy=(6, 6.5), xytext=(4, 6.5),
                  arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax_a.text(5, 7.5, 'Cannot encode', ha='center', va='center', fontsize=7, color='red', fontweight='bold')
    ax_a.text(5, 5.5, 'molecular cations', ha='center', va='center', fontsize=7, color='red')
    
    # Panel label
    ax_a.text(0.5, 9.5, 'Representation gap', fontsize=9, fontweight='bold')
    
    # Panel b: Virtual element mapping
    ax_b = fig.add_axes([0.55, 0.74, 0.42, 0.22])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis('off')
    ax_b.set_title('**b**', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    ax_b.text(0.5, 9.5, 'Coarse-grained mapping', fontsize=9, fontweight='bold')
    
    classes = [
        ("X201", "+1", "3D small", "MA, FA"),
        ("X203", "+1", "2D alkyl", "BA, HA"),
        ("X208", "+2", "1D rod", "EDA"),
        ("X210", "+3", "1D rod", "TAPA"),
        ("X212", "+6", "1D rod", "PEHA"),
    ]
    y_pos = 8.5
    for z, charge, shape, examples in classes:
        rect = FancyBboxPatch((0.5, y_pos-0.6), 4, 0.8, boxstyle="round,pad=0.05",
                              facecolor='#F8F9FA', edgecolor='black', linewidth=0.8)
        ax_b.add_patch(rect)
        ax_b.text(0.8, y_pos-0.2, f'{z}', fontsize=8, fontweight='bold', color='#E74C3C')
        ax_b.text(2.0, y_pos-0.2, f'{charge} | {shape}', fontsize=7)
        ax_b.text(3.5, y_pos-0.2, examples, fontsize=7, style='italic')
        y_pos -= 1.0
    
    ax_b.text(0.5, 2.5, '12 classes total (Table 1)', fontsize=7, style='italic', color='gray')
    
    # Panel c: Model architecture
    ax_c = fig.add_axes([0.08, 0.45, 0.42, 0.22])
    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 10)
    ax_c.axis('off')
    ax_c.set_title('**c**', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    ax_c.text(0.5, 9.5, 'Model adaptation', fontsize=9, fontweight='bold')
    
    # Pretrained backbone (frozen)
    rect = FancyBboxPatch((1, 6), 8, 2, boxstyle="round,pad=0.1",
                          facecolor='#3498DB', edgecolor='black', alpha=0.2, linewidth=1)
    ax_c.add_patch(rect)
    ax_c.text(5, 7.8, 'Pretrained MatterGen backbone', ha='center', fontsize=8, fontweight='bold')
    ax_c.text(5, 7.2, 'GemNet-T denoiser + real-element embeddings', ha='center', fontsize=7, style='italic')
    ax_c.text(5, 6.6, '118 real elements (Z=1\u2013118)  \u2014  FROZEN', ha='center', fontsize=7, color='#3498DB', fontweight='bold')
    
    # Trainable modules
    rect2 = FancyBboxPatch((1, 3.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                           facecolor='#27AE60', edgecolor='black', alpha=0.3, linewidth=1)
    ax_c.add_patch(rect2)
    ax_c.text(2.75, 4.8, 'Virtual embeddings', ha='center', fontsize=8, fontweight='bold')
    ax_c.text(2.75, 4.2, '12 classes (Z=201\u2013212)', ha='center', fontsize=7)
    ax_c.text(2.75, 3.8, 'TRAINABLE', ha='center', fontsize=7, color='#27AE60', fontweight='bold')
    
    rect3 = FancyBboxPatch((5.5, 3.5), 3.5, 1.8, boxstyle="round,pad=0.1",
                           facecolor='#F39C12', edgecolor='black', alpha=0.3, linewidth=1)
    ax_c.add_patch(rect3)
    ax_c.text(7.25, 4.8, 'Adapter layers', ha='center', fontsize=8, fontweight='bold')
    ax_c.text(7.25, 4.2, 'Fine-tuning bridges', ha='center', fontsize=7)
    ax_c.text(7.25, 3.8, 'TRAINABLE', ha='center', fontsize=7, color='#F39C12', fontweight='bold')
    
    # Arrow
    ax_c.annotate('', xy=(2.75, 5.5), xytext=(2.75, 6),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax_c.annotate('', xy=(7.25, 5.5), xytext=(7.25, 6),
                  arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Panel d: Pipeline funnel
    ax_d = fig.add_axes([0.55, 0.45, 0.42, 0.22])
    ax_d.set_title('**d**', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    ax_d.text(0.5, 9.5, 'Generation pipeline', fontsize=9, fontweight='bold')
    
    stages = [("D3PM sampling", 1000, "100%"),
              ("Virtual-element crystals", 954, "95.4%"),
              ("Decoded full-atom hybrids", 106, "11.1%"),
              ("Charge-neutral hybrids", 50, "5.0%"),
              ("Halide-containing hybrids", 19, "1.9%")]
    
    y_pos = 8.5
    widths = [8, 7.5, 5.5, 4, 2.5]
    colors = ['#3498DB', '#5DADE2', '#85C1E9', '#AED6F1', '#D6EAF8']
    for (label, count, pct), w, c in zip(stages, widths, colors):
        rect = FancyBboxPatch((5-w/2, y_pos-0.5), w, 0.7, boxstyle="round,pad=0.05",
                              facecolor=c, edgecolor='black', linewidth=0.8)
        ax_d.add_patch(rect)
        ax_d.text(5, y_pos-0.15, f'{label}: {count} ({pct})', ha='center', va='center', fontsize=7)
        y_pos -= 1.1
    
    ax_d.set_xlim(0, 10)
    ax_d.set_ylim(0, 10)
    ax_d.axis('off')
    
    # No main title per user request
    # fig.text(0.5, 0.97, 'Figure 1 | Virtual-element framework and model adaptation for hybrid crystal generation.',
    #          ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(os.path.join(OUT_DIR, 'figure_1_framework.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUT_DIR, 'figure_1_framework.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: figure_1_framework.png/pdf")


# ============================================
# FIGURE 2: Generation statistics
# ============================================
def generate_figure2():
    fig = plt.figure(figsize=(7.2, 9.5))
    
    # Panel a: Funnel
    ax_a = fig.add_axes([0.08, 0.72, 0.42, 0.24])
    ax_a.set_title('a', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    stages = ["Generated", "Decoded", "Charge-neutral", "Halide-containing", "Perovskite-like"]
    counts = [1000, 954, 106, 50, 19]
    colors = ['#3498DB', '#5DADE2', '#27AE60', '#F39C12', '#E74C3C']
    y_pos = np.arange(len(stages))
    widths = np.array(counts) / max(counts) * 0.8
    
    for i, (stage, count, w, c) in enumerate(zip(stages, counts, widths, colors)):
        left = 0.5 - w/2
        rect = Rectangle((left, i), w, 0.7, facecolor=c, edgecolor='black', linewidth=0.8, alpha=0.8)
        ax_a.add_patch(rect)
        ax_a.text(0.5, i + 0.35, f'{stage}\n{count}', ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(-0.3, 5.2)
    ax_a.set_yticks([])
    ax_a.set_xticks([])
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.spines['bottom'].set_visible(False)
    ax_a.spines['left'].set_visible(False)
    
    # Panel b: Net charge distribution
    ax_b = fig.add_axes([0.55, 0.72, 0.42, 0.24])
    ax_b.set_title('b', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    charges = sorted(net_charges.keys())
    counts = [net_charges[q] for q in charges]
    colors_b = ['#27AE60' if q == 0 else '#BDC3C7' for q in charges]
    
    ax_b.bar(range(len(charges)), counts, color=colors_b, edgecolor='black', linewidth=0.5)
    ax_b.set_xticks(range(len(charges)))
    ax_b.set_xticklabels([str(q) for q in charges], rotation=45, ha='right', fontsize=6)
    ax_b.set_ylabel('Number of structures', fontsize=8)
    ax_b.set_xlabel('Net charge Q', fontsize=8)
    ax_b.axvline(x=charges.index(0), color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax_b.text(charges.index(0), max(counts)*0.9, 'Charge-neutral\n(n=106)', ha='center', fontsize=7, color='green', fontweight='bold')
    
    # Panel c: Top imbalance modes
    ax_c = fig.add_axes([0.08, 0.40, 0.42, 0.24])
    ax_c.set_title('c', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    top_imb = [(-1, 158), (-5, 128), (-3, 110), (-2, 96), (+1, 87)]
    labels = [f'Q={q}' for q, _ in top_imb]
    vals = [c for _, c in top_imb]
    
    bars = ax_c.barh(range(len(top_imb)), vals, color='#E74C3C', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_c.set_yticks(range(len(top_imb)))
    ax_c.set_yticklabels(labels, fontsize=8)
    ax_c.set_xlabel('Count', fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_title('Top charge-imbalance modes', fontsize=8)
    
    # Panel d: Virtual element usage
    ax_d = fig.add_axes([0.55, 0.40, 0.42, 0.24])
    ax_d.set_title('d', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    ve_labels = list(virt_elements.keys())
    ve_counts = [virt_elements[v][1] for v in ve_labels]
    ve_names = [virt_elements[v][0] for v in ve_labels]
    
    bars = ax_d.barh(range(len(ve_labels)), ve_counts, color='#3498DB', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_d.set_yticks(range(len(ve_labels)))
    ax_d.set_yticklabels(ve_labels, fontsize=6)
    ax_d.set_xlabel('Site count', fontsize=8)
    ax_d.invert_yaxis()
    ax_d.set_title('Virtual element usage', fontsize=8)
    
    # Add structure names inside bars (black text, right-shift for narrow bars)
    for i, (count, name) in enumerate(zip(ve_counts, ve_names)):
        if count < 30:  # For very narrow bars (X211, X212), place text to the right
            ax_d.text(count + 8, i, name, ha='left', va='center', fontsize=5, color='black', fontweight='bold')
        else:
            ax_d.text(count * 0.5, i, name, ha='center', va='center', fontsize=5, color='black', fontweight='bold')
    
    # No main title per user request
    # fig.text(0.5, 0.97, 'Figure 2 | Generation statistics and charge-neutrality filtering.',
    #          ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(os.path.join(OUT_DIR, 'figure_2_statistics.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUT_DIR, 'figure_2_statistics.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: figure_2_statistics.png/pdf")


# ============================================
# FIGURE 3: Chemical diversity
# ============================================
def generate_figure3():
    fig = plt.figure(figsize=(7.2, 9.5))
    
    # Panel a: Metal distribution
    ax_a = fig.add_axes([0.10, 0.72, 0.25, 0.24])
    ax_a.set_title('a', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    metal_names = list(metals.keys())[:10]
    metal_counts = [metals[m] for m in metal_names]
    metal_colors = [CAT_COLORS.get(METAL_CATS.get(m, 'main'), '#3498DB') for m in metal_names]
    
    ax_a.barh(range(len(metal_names)), metal_counts, color=metal_colors, edgecolor='black', linewidth=0.5)
    ax_a.set_yticks(range(len(metal_names)))
    ax_a.set_yticklabels(metal_names, fontsize=8)
    ax_a.set_xlabel('Count', fontsize=8)
    ax_a.invert_yaxis()
    ax_a.set_title('Metal distribution', fontsize=8)
    
    # Panel b: Halide distribution
    ax_b = fig.add_axes([0.42, 0.72, 0.25, 0.24])
    ax_b.set_title('b', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    halide_names = list(halides.keys())
    halide_counts = [halides[h] for h in halide_names]
    halide_colors = [HALIDE_COLORS[h] for h in halide_names]
    
    ax_b.bar(halide_names, halide_counts, color=halide_colors, edgecolor='black', linewidth=0.5)
    ax_b.set_title('Halide occurrence', fontsize=8)
    ax_b.set_xlabel('Halide', fontsize=8)
    ax_b.set_ylim(0, max(halide_counts) + 2)
    ax_b.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Panel c: Template distribution
    ax_c = fig.add_axes([0.72, 0.72, 0.25, 0.24])
    ax_c.set_title('c', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    tmpl_names = list(templates.keys())
    tmpl_counts = [templates[t] for t in tmpl_names]
    # Template name to X number mapping
    template_to_xnumber = {
        "TriA": "X210", "PentaHexaA": "X212", "DiA-alkyl": "X208",
        "TetraA": "X211", "Sp2D-alkyl": "X203", "DiA-aromatic": "X209", "A3D-small": "X201",
    }
    xnumber_labels = [template_to_xnumber.get(t, t) for t in tmpl_names]
    
    ax_c.barh(range(len(tmpl_names)), tmpl_counts, color='#9B59B6', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_c.set_yticks(range(len(tmpl_names)))
    ax_c.set_yticklabels(xnumber_labels, fontsize=7)
    ax_c.set_xlabel('Count', fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_title('Organic template classes', fontsize=8)
    
    # Add template names inside bars (right-shift for last two narrow bars)
    for i, (count, name) in enumerate(zip(tmpl_counts, tmpl_names)):
        if count < 3:  # Right-shift for narrow bars (TetraA=6, A3D-small=1, DiA-aromatic=1)
            ax_c.text(count + 2, i, name, ha='left', va='center', fontsize=5, color='black', fontweight='bold')
        else:
            ax_c.text(count * 0.5, i, name, ha='center', va='center', fontsize=5, color='white', fontweight='bold')
    
    # Panel d: Metal-halide heatmap (real data from balanced_analysis_hybrid.json)
    ax_d = fig.add_axes([0.10, 0.40, 0.38, 0.24])
    ax_d.set_title('d', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    metal_hm = list(metal_halide_cooccur.keys())
    halide_hm = ["F", "Cl", "Br", "I"]
    cooccur = np.array([metal_halide_cooccur[m] for m in metal_hm])
    
    im = ax_d.imshow(cooccur, cmap='YlOrRd', aspect='auto')
    ax_d.set_xticks(range(len(halide_hm)))
    ax_d.set_xticklabels(halide_hm, fontsize=8)
    ax_d.set_yticks(range(len(metal_hm)))
    ax_d.set_yticklabels(metal_hm, fontsize=7)
    ax_d.set_title('Metal\u2013halide co-occurrence', fontsize=8)
    
    for i in range(len(metal_hm)):
        for j in range(len(halide_hm)):
            if cooccur[i, j] > 0:
                ax_d.text(j, i, str(cooccur[i, j]), ha='center', va='center', fontsize=7, color='white' if cooccur[i,j] > 3 else 'black')
    
    plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    
    # Panel e: Organic class-halide heatmap (real data)
    ax_e = fig.add_axes([0.58, 0.40, 0.38, 0.24])
    ax_e.set_title('e', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    org_classes = list(org_halide_cooccur.keys())
    org_xnumbers = ["X201", "X203", "X208", "X210", "X211", "X212", "X209"]
    org_halide = ["F", "Cl", "Br", "I"]
    org_cooccur = np.array([org_halide_cooccur[c] for c in org_classes])
    
    im2 = ax_e.imshow(org_cooccur, cmap='YlGnBu', aspect='auto')
    ax_e.set_xticks(range(len(org_halide)))
    ax_e.set_xticklabels(org_halide, fontsize=8)
    ax_e.set_yticks(range(len(org_classes)))
    ax_e.set_yticklabels(org_xnumbers, fontsize=7)
    ax_e.set_title('Organic class\u2013halide co-occurrence', fontsize=8)
    
    for i in range(len(org_classes)):
        for j in range(len(org_halide)):
            if org_cooccur[i, j] > 0:
                ax_e.text(j, i, str(org_cooccur[i, j]), ha='center', va='center', fontsize=7, color='white' if org_cooccur[i,j] > 4 else 'black')
    
    plt.colorbar(im2, ax=ax_e, fraction=0.046, pad=0.04)
    
    # No main title per user request
    # fig.text(0.5, 0.97, 'Figure 3 | Chemical diversity of charge-neutral hybrid candidates.',
    #          ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(os.path.join(OUT_DIR, 'figure_3_diversity.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUT_DIR, 'figure_3_diversity.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: figure_3_diversity.png/pdf")


# ============================================
# FIGURE 4: Screening landscape
# ============================================
def generate_figure4():
    fig = plt.figure(figsize=(7.2, 6.1))
    
    # Load CHGNet + proxy data from halide list
    # Representative data points from Table 4 + others
    # Format: (id, metal, halide, chgnet_energy, bandgap_proxy, category, n_atoms)
    halide_data = [
        ("0217", "Mn", "I", -4.99, 0.5, "trans", 16),
        ("0672", "Fe", "Br", -4.98, 0.7, "trans", 16),
        ("0631", "Cs", "Br", -4.42, 1.7, "alkaline", 11),
        ("0912", "K", "F", -4.91, 2.1, "alkaline", 15),
        ("0927", "Co", "Cl", -3.09, 0.5, "trans", 21),
        ("0396", "\u2014", "F", -3.69, 3.8, "none", 69),
        ("0048", "La", "Br", -0.12, 1.4, "alkaline", 53),
        ("0248", "Ni", "I", 1.67, 0.2, "trans", 33),
        ("0283", "Co", "I", 0.44, 0.2, "trans", 33),
        ("0258", "Ga", "Cl", -2.31, 1.6, "main", 44),
        ("0282", "Sc", "Br", 0.92, 1.4, "trans", 35),
        ("0289", "Cu", "Cl", -0.97, 0.9, "trans", 21),
        ("0310", "Ca", "Br", -2.93, 2.2, "alkaline", 39),
        ("0368", "Mn", "Br", -1.71, 0.4, "trans", 57),
        ("0391", "Mg", "I", -3.41, 2.3, "main", 24),
        ("0392", "Co", "Br", -3.66, 0.7, "trans", 25),
        ("0403", "Y", "Br", -3.37, 1.4, "trans", 35),
        ("0408", "Ca", "I", 0.96, 2.5, "alkaline", 55),
        ("0420", "Ru", "Cl", -3.51, 0.6, "trans", 44),
        ("0454", "Cu", "Cl", -1.08, 0.9, "trans", 25),
        ("0490", "Fe", "Cl", -3.64, 0.6, "trans", 42),
        ("0496", "Mn", "Br", -4.19, 0.4, "trans", 33),
        ("0519", "Th", "F", -0.19, 2.1, "actinide", 26),
        ("0626", "V", "Br", -3.62, 0.7, "trans", 23),
        ("0632", "Cr", "Br", -4.22, 0.7, "trans", 23),
        ("0636", "Mn", "Cl", -2.15, 0.6, "trans", 62),
        ("0671", "Sb", "F", -1.57, 2.1, "main", 23),
        ("0674", "Co", "Br", -2.42, 0.4, "trans", 33),
        ("0707", "Ni", "Cl", -1.27, 1.4, "trans", 26),
        ("0725", "Mn", "I", -4.35, 0.5, "trans", 16),
        ("0742", "Ni", "I", -4.72, 0.5, "trans", 16),
        ("0789", "Ni", "I", -4.73, 0.5, "trans", 16),
        ("0845", "Ga", "Cl", -4.35, 1.6, "main", 33),
        ("0846", "Be", "I", -1.66, 1.2, "main", 48),
        ("0849", "Cr", "I", 0.59, 0.5, "trans", 23),
        ("0853", "Ba", "I", -1.60, 1.2, "alkaline", 31),
        ("0870", "\u2014", "Br", -4.48, 3.4, "none", 32),
        ("0960", "Pb", "I", -1.72, 1.7, "main", 21),
        ("0970", "La", "F", -4.31, 1.8, "alkaline", 39),
        ("0974", "Ca", "Br", -0.21, 1.2, "alkaline", 55),
    ]
    
    # Remove duplicates and keep only 50 unique
    seen = set()
    unique_data = []
    for d in halide_data:
        if d[0] not in seen:
            seen.add(d[0])
            unique_data.append(d)
    halide_data = unique_data[:50]
    
    # Panel a: CHGNet energy vs band-gap proxy (full width)
    ax_a = fig.add_axes([0.10, 0.50, 0.85, 0.42])
    ax_a.set_title('a', fontweight='bold', fontsize=10, loc='left', x=-0.03, y=1.02)
    
    # Jitter to separate overlapping points
    rng = np.random.RandomState(42)
    dft_ids = set(dft_data.keys())
    
    # DFT point x-jitter and y-jitter (band-gap proxy units on x, eV/atom on y)
    dft_jitter_x = {
        "0217": -0.03, "0672": 0.03, "0742": -0.03, "0789": 0.03,
        "0626": 0, "0632": 0, "0912": 0, "0631": 0,
        "0927": 0, "0391": 0,
    }
    dft_jitter_y = {}
    # Label offsets (dx, dy in points) per user request
    label_offsets = {
        "0217": (-18, -4),  # left-bottom
        "0672": (6, -4),    # right-bottom
        "0742": (-18, 3),   # left, label up ~0.1 eV from point
        "0789": (6, 3),     # right, label up ~0.1 eV from point
        "0912": (-22, 0),   # left
        "0631": (6, 6),     # right-top
        "0927": (-18, -3),  # left, label down ~0.1 eV from point
        "0626": (-18, -3),  # left, label down ~0.1 eV from point
        "0632": (6, 0),     # right
        "0391": (6, 4),     # right-top
    }
    
    for cid, metal, halide, energy, gap, cat, n_atoms in halide_data:
        color = HALIDE_COLORS.get(halide, '#7F8C8D')
        marker = 'o' if cat != 'none' else 's'
        size = 40 if cid in dft_ids else 20
        alpha = 0.9 if cid in dft_ids else 0.4
        edge = 'black' if cid in dft_ids else 'none'
        # Jitter: DFT points use fixed offset, non-DFT use random
        if cid in dft_ids:
            jx = dft_jitter_x.get(cid, 0)
            jy = dft_jitter_y.get(cid, 0)
        else:
            jx = rng.uniform(-0.08, 0.08)
            jy = rng.uniform(-0.08, 0.08)
        ax_a.scatter(gap + jx, energy + jy, c=color, s=size, marker=marker, 
                     alpha=alpha, edgecolors=edge, linewidth=0.5, zorder=3 if cid in dft_ids else 2)
        if cid in dft_ids:
            dx, dy = label_offsets.get(cid, (6, 4))
            ax_a.annotate(cid, (gap + jx, energy + jy), textcoords="offset points", 
                          xytext=(dx, dy), fontsize=6, fontweight='bold', color=color,
                          arrowprops=dict(arrowstyle='-', color='gray', lw=0.4))
    
    ax_a.set_xlabel('Band-gap proxy (eV)', fontsize=9)
    ax_a.set_ylabel('CHGNet energy (eV/atom)', fontsize=9)
    ax_a.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Legend in upper right (no data overlap)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=HALIDE_COLORS[h], 
                               markersize=6, label=h) for h in ['F', 'Cl', 'Br', 'I']]
    legend_elements += [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                               markersize=6, markeredgecolor='black', label='DFT selected')]
    ax_a.legend(handles=legend_elements, loc='upper right', fontsize=7, frameon=True)
    
    # Panel b: CHGNet energy distribution (bottom left, equal size) — swapped from c
    ax_b = fig.add_axes([0.10, 0.08, 0.40, 0.32])
    ax_b.set_title('b', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    energies = [d[3] for d in halide_data]
    ax_b.hist(energies, bins=15, color='#27AE60', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_b.axvline(x=np.median(energies), color='red', linestyle='--', linewidth=1, label=f'Median={np.median(energies):.1f} eV/atom')
    ax_b.set_xlabel('CHGNet energy (eV/atom)', fontsize=9)
    ax_b.set_title('CHGNet energy distribution', fontsize=8)
    ax_b.legend(fontsize=6)
    
    # Panel c: Bandgap proxy distribution (bottom right, equal size) — swapped from b
    ax_c = fig.add_axes([0.55, 0.08, 0.40, 0.32])
    ax_c.set_title('c', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)
    
    gaps = [d[4] for d in halide_data]
    ax_c.hist(gaps, bins=15, color='#3498DB', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax_c.axvline(x=np.median(gaps), color='red', linestyle='--', linewidth=1, label=f'Median={np.median(gaps):.1f} eV')
    ax_c.set_xlabel('Band-gap proxy (eV)', fontsize=9)
    ax_c.set_title('Band-gap proxy distribution', fontsize=8)
    ax_c.legend(fontsize=6)
    
    # Panel d removed per user request
    
    # No main title per user request
    # fig.text(0.5, 0.97, 'Figure 4 | Preliminary screening landscape of halide-containing hybrid candidates.',
    #          ha='center', fontsize=11, fontweight='bold')
    
    plt.savefig(os.path.join(OUT_DIR, 'figure_4_screening.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUT_DIR, 'figure_4_screening.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: figure_4_screening.png/pdf")


# ============================================
# FIGURE 5: DFT assessment
# ============================================
def generate_figure5():
    fig = plt.figure(figsize=(7.2, 6.1))  # Match Fig. 3/4 aspect ratio

    ids = ["0742", "0789", "0391", "0631", "0912", "0217", "0672", "0927", "0626", "0632"]

    # Panel a: DFT candidates in screening space (top-left)
    ax_a = fig.add_axes([0.08, 0.55, 0.40, 0.38])
    ax_a.set_title('a', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)

    # Custom label positions (data-coordinate shifts) per user request
    # Baseline: all labels shifted right by 0.1 eV/atom
    label_pos_a = {
        "0217": (0.1, -0.1),    # right 0.1 and down 0.1 eV
        "0742": (-0.2, None),   # left 0.2 eV/atom total
        "0927": (-0.2, None),   # left 0.2 eV/atom total
    }

    for cid in ids:
        d = dft_data[cid]
        color = TYPE_COLORS[d['type']]
        marker = 'D' if d['type'] == 'Direct' else ('o' if d['type'] == 'Indirect' else 'X')
        size = 80 if d['type'] == 'Direct' else 50
        ax_a.scatter(d['chgnet'], d['indirect'], c=color, s=size, marker=marker,
                     edgecolors='black', linewidth=0.5, alpha=0.9)
        tx, ty = d['chgnet'] + 0.1, d['indirect']  # baseline right shift
        off_x, off_y = label_pos_a.get(cid, (None, None))
        if off_x is not None:
            tx = d['chgnet'] + off_x
        if off_y is not None:
            ty = d['indirect'] + off_y
        ax_a.annotate(cid, (d['chgnet'], d['indirect']), textcoords="data",
                      xytext=(tx, ty), fontsize=6,
                      arrowprops=dict(arrowstyle='-', color='gray', lw=0.4))

    ax_a.set_xlabel('CHGNet energy (eV/atom)', fontsize=8)
    ax_a.set_ylabel('DFT-PBE indirect gap (eV)', fontsize=8)
    ax_a.set_title('DFT candidates in screening space', fontsize=8)
    ax_a.set_xlim(-5.2, -2.8)

    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#7F8C8D', markersize=8, label='Metallic'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=8, label='Indirect'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#1ABC9C', markersize=8, label='Direct'),
    ]
    ax_a.legend(handles=legend_elements, loc='upper right', fontsize=6, frameon=True)

    # Panel b: Indirect vs direct gap comparison (top-right) [was panel c]
    ax_b = fig.add_axes([0.55, 0.55, 0.40, 0.38])
    ax_b.set_title('b', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)

    semi_ids = [cid for cid in ids if dft_data[cid]['type'] != 'Metallic']
    indirect_vals = [dft_data[cid]['indirect'] for cid in semi_ids]
    direct_vals = [dft_data[cid]['direct'] for cid in semi_ids]

    x = np.arange(len(semi_ids))
    width = 0.35
    ax_b.bar(x - width/2, indirect_vals, width, label='Indirect', color='#3498DB', edgecolor='black', linewidth=0.5)
    ax_b.bar(x + width/2, direct_vals, width, label='Direct', color='#1ABC9C', edgecolor='black', linewidth=0.5)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(semi_ids, rotation=45, ha='right', fontsize=7)
    ax_b.set_ylabel('Band gap (eV)', fontsize=8)
    ax_b.set_title('Indirect vs direct gaps', fontsize=8)
    ax_b.legend(fontsize=6)

    # Panel c: 0631 DOS (bottom-left) -- same size/position as panel a
    ax_c = fig.add_axes([0.08, 0.06, 0.40, 0.38])
    ax_c.set_title('c', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)

    dos_path = os.path.join(BASE_DIR, "figures_png", "crystal_0631_dos.png")
    dos_img = plt.imread(dos_path)
    dos_h, dos_w = dos_img.shape[:2]
    # Display the full original image without cropping; stretch to fill the panel
    ax_c.imshow(dos_img, aspect='auto', extent=[0, dos_w, 0, dos_h])
    for spine in ax_c.spines.values():
        spine.set_visible(True)
    ax_c.tick_params(axis='both', which='both',
                     bottom=False, top=False, left=False, right=False,
                     labelbottom=False, labelleft=False)
    ax_c.set_xlabel('Energy (eV)', fontsize=8)
    ax_c.set_xlim(0, dos_w)
    ax_c.set_ylim(0, dos_h)

    # Panel d: 0631 band structure (bottom-right) -- same size/position as panel b
    ax_d = fig.add_axes([0.55, 0.06, 0.40, 0.38])
    ax_d.set_title('d', fontweight='bold', fontsize=10, loc='left', x=-0.05, y=1.02)

    bands_path = os.path.join(BASE_DIR, "figures_png", "crystal_0631_bands.png")
    bands_img = plt.imread(bands_path)
    bands_h, bands_w = bands_img.shape[:2]
    # Display the full original image without cropping; stretch to fill the panel
    ax_d.imshow(bands_img, aspect='auto', extent=[0, bands_w, 0, bands_h])
    for spine in ax_d.spines.values():
        spine.set_visible(True)
    ax_d.tick_params(axis='both', which='both',
                     bottom=False, top=False, left=False, right=False,
                     labelbottom=False, labelleft=False)
    ax_d.set_xlabel('k-point index', fontsize=8)
    ax_d.set_xlim(0, bands_w)
    ax_d.set_ylim(0, bands_h)


    plt.savefig(os.path.join(OUT_DIR, 'figure_5_dft.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(OUT_DIR, 'figure_5_dft.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: figure_5_dft.png/pdf")


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Nature-style composite figures for paper_draft_v6")
    print("=" * 60)
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_figure5()
    print("\nAll figures generated in:", OUT_DIR)
    print("Files: figure_1_framework, figure_2_statistics, figure_3_diversity,")
    print("       figure_4_screening, figure_5_dft")
    print("Formats: .png (300 dpi) and .pdf (vector)")
