"""
Ion functional class taxonomy for compressed virtual element mapping.

Design principle:
  Group organic molecules by crystal-chemical role × charge, not by K-means cluster.
  Each class maps to one virtual element Z, reducing 175→~12 classes.

Taxonomy (12 classes):

  ▸ Monocationic (+1) — used in Org₂MX₄ POC families
    1.  A3D-small       Small A-site cations for 3D perovskite (MA, FA, AZ, GA...)
    2.  A3D-halogenated  Halogenated small A-site (FEA, ClEA, BrEA)
    3.  Sp2D-alkyl      Linear/branched alkyl spacer (BA, HA, OA, DA...)
    4.  Sp2D-aromatic   Aromatic spacer — PEA family (PEA, 4F-PEA, 4Cl-PEA...)
    5.  Sp2D-fused      Fused-ring aromatic spacer (NEA, ANEA, PYEA, TEA...)
    6.  Sp2D-cyclic     Cycloalkyl/adamantyl spacer (CHA, CPA, ADA...)
    7.  Sp2D-functional Functionalized spacer (OEG, cyano, ester, fluoroalkyl...)

  ▸ Dicationic (+2)
    8.  DiA-alkyl       Linear aliphatic diammonium (EDA, PDA, BDA, HDA2...)
    9.  DiA-aromatic    Aromatic diammonium (pXDA, mXDA, oXDA)

  ▸ Tricationic (+3)
   10.  TriA            Triammonium (TAPA, SPMN, TREN, TAB, TMB, TPMA...)

  ▸ Tetracationic (+4)
   11.  TetraA          Tetraammonium (TETA...)

  ▸ Pentacationic or higher (+5,+6)
   12.  PentaHexaA      Penta-/hexaammonium (PEHA...)

Virtual element Z assignment:
  Z = 201 + class_index (0-based), so Z=201..212 for 12 classes.
  Gap region Z=119..200 remains masked.
  Z=213..400 reserved for future expansion.
"""

from dataclasses import dataclass, field
from typing import ClassVar

# ── Virtual element Z offset ─────────────────────────────────────────────────
CLASS_Z_OFFSET = 201  # virtual elements start at Z=201 (gap 119-200 masked)

# ── Ion class definitions ────────────────────────────────────────────────────
@dataclass
class IonClass:
    """Definition of one ion functional class."""
    index: int            # 0-based class index
    z: int                # virtual element Z = 201 + index
    symbol: str           # virtual element symbol, e.g. "X201"
    name: str             # short class name
    description: str      # human-readable description
    charge: int           # net ionic charge (+1, +2, +3, +4, +6)
    typical_radius: float # typical effective radius in Angstrom
    member_abbrs: list[str] = field(default_factory=list)  # cation abbreviations
    possible_charges: list[int] = field(default_factory=list)  # allowed protonation states

    def __post_init__(self):
        if not self.possible_charges:
            self.possible_charges = [self.charge]


# ── Monocationic (+1) ────────────────────────────────────────────────────────
ION_CLASS_A3D_SMALL = IonClass(
    index=0, z=201, symbol="X201",
    name="A3D-small",
    description="Small A-site cation for 3D perovskite (tolerance factor ~0.8-1.0)",
    charge=1, typical_radius=3.0,
    member_abbrs=[
        # Simple ammonium
        "MA", "EA", "PA", "IPA", "IBA", "TBA", "AM",
        # Hydroxyl / ether
        "HA", "HYA", "HOEA", "HOPA",
        # Amidinium / guanidinium
        "FA", "AA", "GA", "AGA", "DGA",
        # Cyclic ammoniums
        "AZ", "PYR", "PIP", "IM", "PYZ",
        # Hydrazinium
        "HZ", "MHZ",
        # Unsaturated
        "AL", "PRG",
        # Cyano / nitro
        "CYA", "NTA", "APN",
        # Dimethyl / trimethyl ammonium
        "DMA", "TMA", "TetraMA", "MIM",
        # Methylhydrazinium / hydroxylammonium dupes
        # (HA/HYA already covered above)
    ],
)

ION_CLASS_A3D_HALOGENATED = IonClass(
    index=1, z=202, symbol="X202",
    name="A3D-halogenated",
    description="Halogenated small A-site cation (electronic tuning via X-substitution)",
    charge=1, typical_radius=3.5,
    member_abbrs=["FEA", "ClEA", "BrEA"],
)

ION_CLASS_SP2D_ALKYL = IonClass(
    index=2, z=203, symbol="X203",
    name="Sp2D-alkyl",
    description="Linear/branched alkylammonium spacer for RP 2D perovskite",
    charge=1, typical_radius=6.0,
    member_abbrs=[
        # Linear alkyl C4-C18
        "BA", "PA5", "HA", "HA7", "OA", "NA", "DA", "UDA", "DDA", "TDA", "HDA", "ODA",
        # Branched alkyl
        "IBA2", "EHA", "DMA2",
        # Unsaturated
        "OLE", "CINA",
        # Fluoroalkyl (terminal F)
        "FBA", "FHA", "FOA", "DFBA",
    ],
)

ION_CLASS_SP2D_AROMATIC = IonClass(
    index=3, z=204, symbol="X204",
    name="Sp2D-aromatic",
    description="Aromatic spacer — PEA/BZA family, tunable via ring substitution",
    charge=1, typical_radius=7.0,
    member_abbrs=[
        # PEA series
        "PEA", "4F-PEA", "3F-PEA", "2F-PEA",
        "4Cl-PEA", "4Br-PEA", "4I-PEA",
        "4Me-PEA", "4OMe-PEA", "4OH-PEA",
        "4CN-PEA", "4NO2-PEA", "4NH2-PEA",
        "3Me-4F-PEA", "34F2-PEA", "345F3-PEA",
        "35F2-PEA", "25F2-PEA", "26F2-PEA", "24F2-PEA",
        "PentaF-PEA", "4CF3-PEA", "4SCN-PEA",
        "3F-PEA2",
        # BZA series
        "BZA", "4F-BZA", "3F-BZA", "4Cl-BZA",
        "4Br-BZA", "4Me-BZA", "4OMe-BZA", "34F2-BZA",
        # Phenylpropyl / phenylbutyl
        "PPA", "PBA", "4F-PPA", "4F-PBA",
    ],
)

ION_CLASS_SP2D_FUSED = IonClass(
    index=4, z=205, symbol="X205",
    name="Sp2D-fused",
    description="Fused-ring aromatic spacer — naphthalene, anthracene, pyrene, heterocyclic",
    charge=1, typical_radius=8.0,
    member_abbrs=[
        # Naphthalene
        "NEA", "NMA", "2-NEA", "2-NMA", "6F-NEA",
        # Biphenyl / terphenyl
        "BPEA", "BPMA",
        # Anthracene / pyrene
        "ANEA", "PYEA",
        # Thiophene
        "TEA", "TMA2", "BTEA", "TTPA",
        # Furan / pyridine
        "FEA", "PyEA", "PyMA", "2PyEA", "3PyEA",
    ],
)

ION_CLASS_SP2D_CYCLIC = IonClass(
    index=5, z=206, symbol="X206",
    name="Sp2D-cyclic",
    description="Cycloalkyl / adamantyl spacer — rigid, non-planar, hydrophobic",
    charge=1, typical_radius=5.5,
    member_abbrs=[
        "CHA", "CPA", "CHMA", "CHEA",
        "ADA", "ADMA", "ADEA",
    ],
)

ION_CLASS_SP2D_FUNCTIONAL = IonClass(
    index=6, z=207, symbol="X207",
    name="Sp2D-functional",
    description="Functionalized spacer — ether, cyano, ester, siloxane, amino-acid derived",
    charge=1, typical_radius=7.0,
    member_abbrs=[
        # Ether
        "MOPA",
        # Oligo(ethylene glycol)
        "OEG1", "OEG2", "OEG3",
        # Cyano-functionalized
        "CNBA", "CNOA",
        # Ester
        "AEBA", "MEHA",
        # Siloxane
        "AMPS",
        # Pyridinium-ethyl ammoniums
        "PyREA", "2PyREA", "3PyREA", "4PyREA",
        # Guanidinium-substituted
        "GABA", "GAPA", "GABUT", "GAPEN", "GAHEX", "AGMAT",
        # Amino acid derived
        "ORN", "LYS", "ARG", "HIS",
        # DABCO / TMG (monocationic)
        "DABCO", "TMGUA",
    ],
)

# ── Dicationic (+2) ──────────────────────────────────────────────────────────
ION_CLASS_DIA_ALKYL = IonClass(
    index=7, z=208, symbol="X208",
    name="DiA-alkyl",
    description="Linear aliphatic diammonium linker (1D/2D Dion-Jacobson phases)",
    charge=2, typical_radius=7.0,
    member_abbrs=[
        # Small (also in A-site list)
        "EDA", "PDA", "BDA",
        # Spacer diammonium
        "BDA2", "PDA2", "HDA2", "ODA2", "DDA2", "DDDA2",
        # Triamine-derived (+2)
        "DPTA", "SPMD", "TACN",
    ],
)

ION_CLASS_DIA_AROMATIC = IonClass(
    index=8, z=209, symbol="X209",
    name="DiA-aromatic",
    description="Aromatic diammonium spacer — rigid, pi-stacking capable",
    charge=2, typical_radius=8.0,
    member_abbrs=["pXDA", "mXDA", "oXDA"],
)

# ── Tricationic (+3) ─────────────────────────────────────────────────────────
ION_CLASS_TRIA = IonClass(
    index=9, z=210, symbol="X210",
    name="TriA",
    description="Triammonium cation — tripodal, cage, or aromatic trication (charge +3)",
    charge=3, typical_radius=8.0,
    member_abbrs=[
        # Aliphatic
        "TAPA", "SPMN", "TREN", "TAPA3", "TAHA",
        # Aromatic
        "TAB", "TABMe", "TABF", "TMB", "TEB",
        # Guanidinium-derived
        "TGUA", "BIGUA",
        # Cyclic / cage
        "CYTA", "CYTT", "ADTA",
        # Tripodal
        "TPMA", "TPEA", "TTMA",
    ],
)

# ── Tetracationic (+4) ───────────────────────────────────────────────────────
ION_CLASS_TETRAA = IonClass(
    index=10, z=211, symbol="X211",
    name="TetraA",
    description="Tetraammonium cation — linear tetramine (TETA-family)",
    charge=4, typical_radius=9.0,
    member_abbrs=["TETA"],
)

# ── Penta- / Hexacationic (+5, +6) ──────────────────────────────────────────
ION_CLASS_PENTA_HEXA_A = IonClass(
    index=11, z=212, symbol="X212",
    name="PentaHexaA",
    description="Penta-/hexaammonium cation — PEHA-family, highly charged",
    charge=6, typical_radius=10.0,
    member_abbrs=["PEHA"],
)

# ── Master list ──────────────────────────────────────────────────────────────
ION_CLASSES: list[IonClass] = [
    ION_CLASS_A3D_SMALL,
    ION_CLASS_A3D_HALOGENATED,
    ION_CLASS_SP2D_ALKYL,
    ION_CLASS_SP2D_AROMATIC,
    ION_CLASS_SP2D_FUSED,
    ION_CLASS_SP2D_CYCLIC,
    ION_CLASS_SP2D_FUNCTIONAL,
    ION_CLASS_DIA_ALKYL,
    ION_CLASS_DIA_AROMATIC,
    ION_CLASS_TRIA,
    ION_CLASS_TETRAA,
    ION_CLASS_PENTA_HEXA_A,
]

# Build lookup tables
ION_CLASS_BY_Z: dict[int, IonClass] = {ic.z: ic for ic in ION_CLASSES}
ION_CLASS_BY_INDEX: dict[int, IonClass] = {ic.index: ic for ic in ION_CLASSES}
ION_CLASS_BY_NAME: dict[str, IonClass] = {ic.name: ic for ic in ION_CLASSES}

# Build abbreviation → IonClass lookup
CATION_TO_CLASS: dict[str, IonClass] = {}
for ic in ION_CLASSES:
    for abbr in ic.member_abbrs:
        CATION_TO_CLASS[abbr] = ic


def get_ion_class_for_cation(abbr: str) -> IonClass | None:
    """Return the IonClass that a cation abbreviation belongs to."""
    return CATION_TO_CLASS.get(abbr)


def get_ion_class_by_z(z: int) -> IonClass | None:
    """Return IonClass for a virtual element Z."""
    return ION_CLASS_BY_Z.get(z)


def get_possible_charges_for_z(z: int) -> list[int]:
    """Return allowed protonation/charge states for a virtual element Z."""
    ic = get_ion_class_by_z(z)
    if ic is not None:
        return ic.possible_charges
    return [1]


def summarize() -> str:
    """Return a human-readable summary of all ion classes."""
    lines = []
    lines.append(f"{'Idx':>3} {'Z':>4} {'Charge':>6} {'Radius':>8} {'Class':<20} {'Members':>7} {'Description'}")
    lines.append("-" * 100)
    for ic in ION_CLASSES:
        lines.append(
            f"{ic.index:3d} {ic.z:4d} {ic.charge:+4d}  "
            f"{ic.typical_radius:6.1f}A  "
            f"{ic.name:<20} {len(ic.member_abbrs):7d} "
            f"{ic.description[:45]}"
        )
    total = sum(len(ic.member_abbrs) for ic in ION_CLASSES)
    lines.append("-" * 100)
    lines.append(f"  Total: {len(ION_CLASSES)} classes covering {total} cations")
    lines.append(f"  Virtual Z range: {ION_CLASSES[0].z} – {ION_CLASSES[-1].z}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())

    # Verify all cations are assigned
    from mattergen.molecule_mapping.cation_library import get_full_cation_library
    lib = get_full_cation_library()
    unassigned = []
    for abbr in lib:
        if abbr not in CATION_TO_CLASS:
            unassigned.append(abbr)

    if unassigned:
        print(f"\n  Unassigned cations ({len(unassigned)}): {unassigned}")
    else:
        print(f"\n  All {len(lib)} cations assigned to classes.")
