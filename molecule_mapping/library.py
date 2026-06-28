# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Virtual Element Library for organic-inorganic hybrid crystal generation.

Maps organic molecules to virtual elements (Z >= VIRTUAL_ELEMENT_OFFSET) using
K-means clustering on physicochemical descriptors (Strategy A from the design doc).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# sklearn imports moved to function-level for environments without sklearn
# (sklearn only needed for K-means strategy; identity mapping works without it)

from mattergen.molecule_mapping.virtual_elements import (
    N_VIRTUAL_ELEMENTS,
    VIRTUAL_ELEMENT_OFFSET,
    get_virtual_element_z,
)
from mattergen.molecule_mapping.descriptors import (
    HAS_RDKIT,
    MolecularDescriptor,
    compute_descriptor_from_smiles,
)


@dataclass
class VirtualElementInfo:
    """Metadata for a single virtual element."""
    z: int                                       # Virtual atomic number (>= VIRTUAL_ELEMENT_OFFSET)
    cluster_center: np.ndarray                   # Cluster center in descriptor space
    representative_smiles: Optional[str] = None  # Most representative SMILES in this cluster
    radius_range: tuple[float, float] = (2.0, 8.0)  # (min, max) effective radius
    charge_distribution: Dict[int, int] = field(default_factory=dict)  # {charge: count}
    n_members: int = 0                           # Number of molecules in this cluster


@dataclass
class VirtualElementLibrary:
    """
    Manages the mapping between organic molecules and virtual elements.

    Strategy A (Discrete clustering):
    - Build K-means clusters on physicochemical descriptors
    - Each cluster = one virtual element
    - New molecules mapped via nearest neighbor

    Attributes:
        n_virtual: Number of virtual elements (= number of clusters)
        cluster_centers: shape (n_virtual, descriptor_dim) - normalized cluster centers
        scaler: Fitted StandardScaler for descriptor normalization
        virtual_to_mols: {virtual_z: [list of SMILES in this cluster]}
        mol_to_virtual: {smiles: virtual_z} - inverse mapping
        virtual_infos: {virtual_z: VirtualElementInfo} - metadata per virtual element
        embeddings: shape (n_virtual, hidden_dim) - optional precomputed embeddings
    """
    n_virtual: int
    cluster_centers: np.ndarray       # (n_virtual, descriptor_dim)
    scaler: object = None  # sklearn.preprocessing.StandardScaler (lazy import)
    virtual_to_mols: Dict[int, List[str]] = field(default_factory=dict)
    mol_to_virtual: Dict[str, int] = field(default_factory=dict)
    virtual_infos: Dict[int, VirtualElementInfo] = field(default_factory=dict)
    embeddings: Optional[torch.Tensor] = None  # (n_virtual, hidden_dim)

    def get_virtual_z(self, smiles: str, charge: int = 1, role: str = "A-site cation") -> int:
        """
        Map a SMILES string to a virtual element atomic number.

        Args:
            smiles: SMILES string of the organic molecule
            charge: Net charge
            role: Crystal lattice role

        Returns:
            Virtual atomic number (>= VIRTUAL_ELEMENT_OFFSET)
        """
        if smiles in self.mol_to_virtual:
            return self.mol_to_virtual[smiles]

        # Compute descriptor and find nearest cluster
        desc = compute_descriptor_from_smiles(smiles, charge=charge, role=role)
        vec = desc.to_vector().reshape(1, -1)
        if self.scaler is not None:
            vec_norm = self.scaler.transform(vec)
        else:
            vec_norm = vec  # No scaler available, use raw features
        distances = np.linalg.norm(self.cluster_centers - vec_norm, axis=1)
        cluster_idx = int(np.argmin(distances))
        return get_virtual_element_z(cluster_idx)

    def get_smiles_from_virtual_z(
        self, virtual_z: int, n_candidates: int = 5
    ) -> List[str]:
        """
        Get candidate SMILES for a virtual element.

        Args:
            virtual_z: Virtual atomic number
            n_candidates: Max number of candidates to return

        Returns:
            List of SMILES strings (best matches first)
        """
        cluster_idx = virtual_z - VIRTUAL_ELEMENT_OFFSET
        if cluster_idx < 0 or cluster_idx >= self.n_virtual:
            return []

        mols = self.virtual_to_mols.get(virtual_z, [])
        return mols[:n_candidates]

    def get_info(self, virtual_z: int) -> Optional[VirtualElementInfo]:
        """Get metadata for a virtual element."""
        return self.virtual_infos.get(virtual_z)

    def get_embedding(self, virtual_z: int, hidden_dim: int) -> torch.Tensor:
        """
        Get the embedding vector for a virtual element.
        If precomputed embeddings are available, use those; otherwise
        use cluster center projected to hidden_dim.

        Args:
            virtual_z: Virtual atomic number
            hidden_dim: Desired embedding dimension

        Returns:
            torch.Tensor of shape (hidden_dim,)
        """
        cluster_idx = virtual_z - VIRTUAL_ELEMENT_OFFSET
        n_centers = len(self.cluster_centers)

        # Return zero vector for out-of-range virtual elements (reserved slots)
        if cluster_idx < 0 or cluster_idx >= n_centers:
            return torch.zeros(hidden_dim, dtype=torch.float32)

        if self.embeddings is not None and self.embeddings.shape[-1] == hidden_dim:
            if 0 <= cluster_idx < self.embeddings.shape[0]:
                return self.embeddings[cluster_idx]

        # Fallback: project cluster center
        center = self.cluster_centers[cluster_idx]
        # Simple repeat/interpolate to hidden_dim
        vec = np.tile(center, hidden_dim // len(center) + 1)[:hidden_dim]
        vec = vec / np.linalg.norm(vec) * math.sqrt(hidden_dim)
        return torch.tensor(vec, dtype=torch.float32)

    def save(self, path: str):
        """Save library to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_virtual": self.n_virtual,
            "cluster_centers": self.cluster_centers.tolist(),
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else [],
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else [],
            "virtual_to_mols": {str(k): v for k, v in self.virtual_to_mols.items()},
            "mol_to_virtual": self.mol_to_virtual,
            "virtual_infos": {
                str(k): {
                    "z": v.z,
                    "cluster_center": v.cluster_center.tolist(),
                    "representative_smiles": v.representative_smiles,
                    "radius_range": list(v.radius_range),
                    "charge_distribution": v.charge_distribution,
                    "n_members": v.n_members,
                }
                for k, v in self.virtual_infos.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "VirtualElementLibrary":
        """Load library from disk."""
        from sklearn.preprocessing import StandardScaler

        with open(path) as f:
            data = json.load(f)

        scaler = StandardScaler()
        if data["scaler_mean"] and data["scaler_scale"]:
            scaler.mean_ = np.array(data["scaler_mean"])
            scaler.scale_ = np.array(data["scaler_scale"])

        lib = cls(
            n_virtual=data["n_virtual"],
            cluster_centers=np.array(data["cluster_centers"]),
            scaler=scaler,
            virtual_to_mols={int(k): v for k, v in data["virtual_to_mols"].items()},
            mol_to_virtual=data["mol_to_virtual"],
        )

        lib.virtual_infos = {
            int(k): VirtualElementInfo(
                z=v["z"],
                cluster_center=np.array(v["cluster_center"]),
                representative_smiles=v.get("representative_smiles"),
                radius_range=tuple(v.get("radius_range", [2.0, 8.0])),
                charge_distribution=v.get("charge_distribution", {}),
                n_members=v.get("n_members", 0),
            )
            for k, v in data.get("virtual_infos", {}).items()
        }

        return lib

    def build_embeddings(self, hidden_dim: int, method: str = "pca_projection"):
        """
        Build embedding vectors for all virtual elements.
        Projects cluster centers into hidden_dim space.

        Args:
            hidden_dim: Target embedding dimension
            method: "pca_projection" or "random_projection"
        """
        if method == "random_projection":
            # Random orthogonal projection
            proj = torch.randn(self.cluster_centers.shape[1], hidden_dim)
            proj, _ = torch.linalg.qr(proj)
            centers_t = torch.tensor(self.cluster_centers, dtype=torch.float32)
            self.embeddings = centers_t @ proj
        else:
            # Simple repeat/interpolate
            d = self.cluster_centers.shape[1]
            repeats = math.ceil(hidden_dim / d)
            centers_t = torch.tensor(self.cluster_centers, dtype=torch.float32)
            self.embeddings = centers_t.repeat_interleave(repeats, dim=1)[:, :hidden_dim]
            # Normalize
            norms = self.embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.embeddings = self.embeddings / norms * math.sqrt(hidden_dim)


def build_virtual_element_library(
    mol_list: List[str],
    charges: Optional[List[int]] = None,
    roles: Optional[List[str]] = None,
    n_virtual: int = N_VIRTUAL_ELEMENTS,
    random_state: int = 42,
) -> VirtualElementLibrary:
    """
    Build a VirtualElementLibrary from a list of SMILES.

    Args:
        mol_list: List of SMILES strings
        charges: Per-molecule charge (default: all +1)
        roles: Per-molecule role (default: all "A-site cation")
        n_virtual: Number of virtual elements (= K for K-means)
        random_state: Random seed for reproducibility

    Returns:
        Configured VirtualElementLibrary
    """
    if charges is None:
        charges = [1] * len(mol_list)
    if roles is None:
        roles = ["A-site cation"] * len(mol_list)

    if n_virtual > len(mol_list):
        n_virtual = max(1, len(mol_list) // 2)
        print(f"[WARN] Reducing n_virtual to {n_virtual} (fewer molecules than requested clusters)")

    # Compute descriptors
    descriptors = []
    for smi, chg, role in zip(mol_list, charges, roles):
        desc = compute_descriptor_from_smiles(smi, charge=chg, role=role)
        descriptors.append(desc)

    # Convert to feature matrix
    X = np.array([d.to_vector() for d in descriptors], dtype=np.float32)

    from sklearn.preprocessing import StandardScaler

    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    # Cluster
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_virtual, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_norm)

    # Build library
    virtual_to_mols: Dict[int, List[str]] = {}
    mol_to_virtual: Dict[str, int] = {}
    virtual_infos: Dict[int, VirtualElementInfo] = {}

    for idx, (smi, label) in enumerate(zip(mol_list, labels)):
        vz = get_virtual_element_z(label)
        if vz not in virtual_to_mols:
            virtual_to_mols[vz] = []
        virtual_to_mols[vz].append(smi)
        mol_to_virtual[smi] = vz

    # Build info per cluster
    for label in range(n_virtual):
        vz = get_virtual_element_z(label)
        mols_in_cluster = virtual_to_mols.get(vz, [])
        # Find cluster members
        member_indices = np.where(labels == label)[0]
        member_descs = [descriptors[i] for i in member_indices]

        if not member_descs:
            continue

        # Representative: closest to cluster center
        center_norm = kmeans.cluster_centers_[label]
        member_vectors = X_norm[member_indices]
        distances = np.linalg.norm(member_vectors - center_norm, axis=1)
        representative_idx = member_indices[int(np.argmin(distances))]
        representative_smiles = mol_list[representative_idx]

        radii = [d.r_eff for d in member_descs]
        charge_dist = {}
        for d in member_descs:
            charge_dist[d.charge] = charge_dist.get(d.charge, 0) + 1

        virtual_infos[vz] = VirtualElementInfo(
            z=vz,
            cluster_center=kmeans.cluster_centers_[label].copy(),
            representative_smiles=representative_smiles,
            radius_range=(min(radii), max(radii)),
            charge_distribution=charge_dist,
            n_members=len(member_descs),
        )

    library = VirtualElementLibrary(
        n_virtual=n_virtual,
        cluster_centers=kmeans.cluster_centers_.copy(),
        scaler=scaler,
        virtual_to_mols=virtual_to_mols,
        mol_to_virtual=mol_to_virtual,
        virtual_infos=virtual_infos,
    )

    return library


# Pre-built known hybrid perovskite cations with their SMILES
KNOWN_PEROVSKITE_CATIONS = {
    "MA":  {"smiles": "C[NH3+]",   "charge": 1, "role": "A-site cation", "r_eff": 2.17},
    "FA":  {"smiles": "C(=[NH2+])N", "charge": 1, "role": "A-site cation", "r_eff": 2.53},
    "EA":  {"smiles": "CC[NH3+]",  "charge": 1, "role": "A-site cation", "r_eff": 2.30},
    "DMA": {"smiles": "C[NH2+]C",  "charge": 1, "role": "A-site cation", "r_eff": 2.50},
    "GA":  {"smiles": "N=C(N)N",   "charge": 1, "role": "A-site cation", "r_eff": 2.78},
    "TMA": {"smiles": "C[NH+](C)C", "charge": 1, "role": "A-site cation", "r_eff": 2.85},
    "PEA": {"smiles": "NCCC1=CC=CC=C1", "charge": 1, "role": "spacer cation", "r_eff": 4.50},
    "BA":  {"smiles": "CCCC[NH3+]", "charge": 1, "role": "spacer cation", "r_eff": 4.20},
    "NEA": {"smiles": "NCCC1=C2C=CC=CC2=CC=C1", "charge": 1, "role": "spacer cation", "r_eff": 6.00},
}

def build_identity_library(
    mol_list: List[str],
    charges: Optional[List[int]] = None,
    roles: Optional[List[str]] = None,
) -> VirtualElementLibrary:
    """
    Build a 1:1 identity library: each molecule gets its own virtual element (no clustering).

    Args:
        mol_list: List of SMILES strings
        charges: Per-molecule charge (default: all +1)
        roles: Per-molecule role (default: all "A-site cation")

    Returns:
        VirtualElementLibrary with n_virtual == len(mol_list)
    """
    if charges is None:
        charges = [1] * len(mol_list)
    if roles is None:
        roles = ["A-site cation"] * len(mol_list)

    # Compute descriptors
    descriptors = []
    for smi, chg, role in zip(mol_list, charges, roles):
        desc = compute_descriptor_from_smiles(smi, charge=chg, role=role)
        descriptors.append(desc)

    X = np.array([d.to_vector() for d in descriptors], dtype=np.float32)

    # Lazy import with fallback (identity library doesn't critically need scaling)
    try:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    except ImportError:
        # Simple numpy-based standardization fallback
        class _SimpleScaler:
            def __init__(self):
                self.mean_ = None
                self.scale_ = None
            def fit_transform(self, X):
                self.mean_ = X.mean(axis=0)
                self.scale_ = X.std(axis=0)
                self.scale_[self.scale_ < 1e-8] = 1.0
                return (X - self.mean_) / self.scale_
            def transform(self, X):
                return (X - self.mean_) / self.scale_
        scaler = _SimpleScaler()

    X_norm = scaler.fit_transform(X)

    n = len(mol_list)

    # 1:1 mapping
    virtual_to_mols: Dict[int, List[str]] = {}
    mol_to_virtual: Dict[str, int] = {}
    virtual_infos: Dict[int, VirtualElementInfo] = {}

    for i, (smi, desc) in enumerate(zip(mol_list, descriptors)):
        vz = get_virtual_element_z(i)
        virtual_to_mols[vz] = [smi]
        mol_to_virtual[smi] = vz
        virtual_infos[vz] = VirtualElementInfo(
            z=vz,
            cluster_center=X_norm[i].copy(),
            representative_smiles=smi,
            radius_range=(desc.r_eff, desc.r_eff),
            charge_distribution={desc.charge: 1},
            n_members=1,
        )

    return VirtualElementLibrary(
        n_virtual=n,
        cluster_centers=X_norm.copy(),
        scaler=scaler,
        virtual_to_mols=virtual_to_mols,
        mol_to_virtual=mol_to_virtual,
        virtual_infos=virtual_infos,
    )


def build_quick_perovskite_library(n_virtual: int = 20) -> VirtualElementLibrary:
    """
    Build a 1:1 identity virtual element library from known perovskite cations.
    Each known cation gets its own unique virtual element (no clustering).

    Uses the comprehensive cation library (~141 cations total) from cation_library.py.

    Args:
        n_virtual: Ignored in identity mode; always uses all known cations

    Returns:
        VirtualElementLibrary with 1:1 molecule-to-virtual-element mapping
    """
    from mattergen.molecule_mapping.cation_library import get_full_cation_library

    full_lib = get_full_cation_library()
    mol_list = [info["smiles"] for info in full_lib.values()]
    charges = [info["charge"] for info in full_lib.values()]
    roles = [info["role"] for info in full_lib.values()]

    return build_identity_library(
        mol_list=mol_list,
        charges=charges,
        roles=roles,
    )
