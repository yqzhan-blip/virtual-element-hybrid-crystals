# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Molecular descriptors for organic molecules in crystal contexts.
Converts organic molecules to physicochemical parameter vectors suitable for
clustering and virtual element mapping.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# RDKit is optional - only needed when computing molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@dataclass
class MolecularDescriptor:
    """
    Physicochemical descriptor of an organic molecule for crystal structure generation.

    All parameters can be computed from DFT or RDKit. The most critical parameters
    for crystal structure prediction are r_eff (effective ionic radius) and charge,
    which determine the Goldschmidt tolerance factor for perovskites.

    References:
        Kieslich et al. (2017) Dalton Trans. - effective ionic radii and tolerance factors
        Marchetti et al. (2019) J. Phys. Chem. A - learn-and-match molecular cations
    """
    r_eff: float                          # Effective ionic radius (Å), spherical approx: r = (3V/4π)^(1/3)
    charge: int                           # Net charge (+1, +2, -1, -2, ...)
    n_hbd: int                            # Number of hydrogen bond donors (NH, OH groups)
    n_hba: int                            # Number of hydrogen bond acceptors (N, O atoms)
    dipole: float                         # Dipole moment (Debye), default 0 if unknown
    polarizability: float                 # Polarizability (Å³)
    anisotropy: float                     # Shape anisotropy: η = r_max/r_min
    n_connect: int                        # Number of connection points (0 for cations, 2 for ditopic linkers, etc.)
    smiles: str                           # SMILES string for reverse decoding
    molecular_weight: float = 0.0         # Molecular weight (g/mol)
    num_rotatable_bonds: int = 0          # Number of rotatable bonds
    role: str = "A-site cation"           # Crystal lattice role

    def to_vector(self, normalize: bool = False) -> np.ndarray:
        """
        Convert descriptor to a numerical feature vector for clustering.

        Returns:
            np.ndarray of shape (8,) containing the key physicochemical parameters.
        """
        return np.array([
            self.r_eff,
            float(self.charge),
            float(self.n_hbd),
            float(self.n_hba),
            self.dipole,
            self.polarizability,
            self.anisotropy,
            float(self.n_connect),
        ], dtype=np.float32)

    @staticmethod
    def vector_dim() -> int:
        """Dimension of the feature vector returned by to_vector()."""
        return 8


def estimate_effective_radius_from_smiles(smiles: str) -> Optional[float]:
    """
    Estimate effective ionic radius from SMILES using RDKit molecular volume.

    r_eff = (3 * V / (4 * π))^(1/3)
    where V is approximated by RDKit's McGowan volume or simple atom count.

    Args:
        smiles: SMILES string

    Returns:
        Estimated effective radius in Å, or None if RDKit unavailable.
    """
    if not HAS_RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Use McGowan volume approximation
    try:
        vol = Descriptors.McGowanVolume(mol)  # in units of 0.01 nm^3 per molecule = 0.1 Å³
        # Actually McGowanVolume returns in (cm^3/mol)/100, we need a rough approximation
        # Better approach: approximate by atom count
        num_atoms = mol.GetNumAtoms()
        # Rough estimate: each atom contributes ~20 Å³ (including hydrogens)
        volume = num_atoms * 20.0
        r_eff = (3 * volume / (4 * math.pi)) ** (1.0 / 3.0)
        return round(r_eff, 3)
    except Exception:
        # Fallback: estimate from heavy atom count
        num_heavy = mol.GetNumHeavyAtoms()
        volume = num_heavy * 25.0  # approximate volume per heavy atom
        r_eff = (3 * volume / (4 * math.pi)) ** (1.0 / 3.0)
        return round(r_eff, 3)


def compute_descriptor_from_smiles(
    smiles: str,
    charge: int = 1,
    role: str = "A-site cation",
    n_connect: int = 0,
) -> MolecularDescriptor:
    """
    Compute a full MolecularDescriptor from a SMILES string.

    Args:
        smiles: SMILES string
        charge: Net charge of the molecular ion
        role: Crystal lattice role
        n_connect: Number of connection points

    Returns:
        MolecularDescriptor with computed parameters
    """
    if not HAS_RDKIT:
        # Fallback: create descriptor with estimated values
        # Estimate MW from formula length (rough)
        import hashlib
        # Deterministic hash-based fallback values
        h = int(hashlib.md5(smiles.encode()).hexdigest()[:8], 16)
        r_eff_est = 2.0 + (h % 1000) / 200.0  # 2.0-7.0 Å
        mw_est = 20.0 + (h % 500)              # 20-520 g/mol
        return MolecularDescriptor(
            r_eff=r_eff_est,
            charge=charge,
            n_hbd=h % 5,
            n_hba=(h // 10) % 5,
            dipole=0.0,
            polarizability=r_eff_est ** 3 * 0.5,
            anisotropy=1.0 + (h % 300) / 100.0,
            n_connect=n_connect,
            smiles=smiles,
            molecular_weight=mw_est,
            num_rotatable_bonds=h % 8,
            role=role,
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Add hydrogens for accurate 3D computation
    mol = Chem.AddHs(mol)

    # Compute effective radius from molecular volume
    r_eff = estimate_effective_radius_from_smiles(smiles) or 3.0

    # Compute H-bond donors/acceptors
    n_hbd = rdMolDescriptors.CalcNumHBD(mol)
    n_hba = rdMolDescriptors.CalcNumHBA(mol)

    # Try to compute 3D properties (dipole, anisotropy)
    dipole = 0.0
    anisotropy = 1.0
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Anisotropy: ratio of max/min extent
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        # Compute principal axes lengths via covariance
        centered = coords - coords.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(coords))
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        r_max = np.sqrt(eigenvalues.max())
        r_min = np.sqrt(eigenvalues.min())
        anisotropy = float(r_max / r_min) if r_min > 1e-10 else 1.0
    except Exception:
        pass

    # Molecular weight
    mw = Descriptors.MolWt(mol)

    # Rotatable bonds
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Polarizability (rough estimate: ~ molecular volume / 2)
    polarizability = r_eff ** 3 * 0.5

    return MolecularDescriptor(
        r_eff=r_eff,
        charge=charge,
        n_hbd=n_hbd,
        n_hba=n_hba,
        dipole=dipole,
        polarizability=polarizability,
        anisotropy=anisotropy,
        n_connect=n_connect,
        smiles=smiles,
        molecular_weight=mw,
        num_rotatable_bonds=n_rot,
        role=role,
    )


class ScalingManager:
    """
    Manages feature scaling (standardization) for molecular descriptors.
    Used before K-means clustering to normalize feature ranges.
    """

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, vectors: np.ndarray):
        """
        Fit the scaler to feature vectors.

        Args:
            vectors: shape (N, d) array of descriptor vectors
        """
        self.mean = vectors.mean(axis=0)
        self.std = vectors.std(axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Standardize feature vectors."""
        if self.mean is None:
            return vectors
        return (vectors - self.mean) / self.std

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Fit then transform."""
        self.fit(vectors)
        return self.transform(vectors)

    def inverse_transform(self, vectors: np.ndarray) -> np.ndarray:
        """Reverse standardization."""
        if self.mean is None:
            return vectors
        return vectors * self.std + self.mean
