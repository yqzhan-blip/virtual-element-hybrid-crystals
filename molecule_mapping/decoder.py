# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Decoder module: converts coarse-grained virtual-element crystals back to
full-atom organic-inorganic hybrid structures.

Steps:
1. For each virtual element site, identify candidate molecule SMILES
2. Select best molecule based on local environment (cavity size, coordination)
3. Generate 3D conformer and orient it in the crystal lattice
"""

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Element, Lattice, PeriodicSite, Species, Structure

from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM,
    VIRTUAL_ELEMENT_OFFSET,
    is_virtual_element,
    get_virtual_element_index,
    get_virtual_symbol,
)
from mattergen.molecule_mapping.library import VirtualElementLibrary
from mattergen.molecule_mapping.descriptors import (
    HAS_RDKIT,
    MolecularDescriptor,
    compute_descriptor_from_smiles,
)

# RDKit optional imports
if HAS_RDKIT:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors


class LocalEnvironment:
    """
    Describes the local environment around a virtual element site in the crystal.

    Used to select the most appropriate organic molecule from candidate list.
    """

    def __init__(
        self,
        position: np.ndarray,
        cavity_radius: float,
        nearest_inorganic_species: List[str],
        nearest_distances: List[float],
        coordination_number: int,
    ):
        self.position = position
        self.cavity_radius = cavity_radius          # Estimated radius of available space (Å)
        self.nearest_inorganic_species = nearest_inorganic_species  # e.g., ["Pb", "I", "I"]
        self.nearest_distances = nearest_distances
        self.coordination_number = coordination_number

    def __repr__(self):
        return (
            f"LocalEnvironment(cavity_r={self.cavity_radius:.2f}Å, "
            f"coord={self.coordination_number}, neighbors={self.nearest_inorganic_species})"
        )


def estimate_cavity_radius(
    virtual_site: PeriodicSite,
    structure: Structure,
    inorganic_atomic_numbers: List[int],
    min_neighbors: int = 6,
) -> Tuple[float, List[str], List[float], int]:
    """
    Estimate the cavity radius around a virtual element site.

    Strategy:
    1. Find nearest N inorganic neighbors
    2. Cavity radius = min(distance to neighbor) - ion_radius(neighbor)
       where ion_radius is estimated from element type

    Args:
        virtual_site: The virtual element site
        structure: Full structure (real + virtual)
        inorganic_atomic_numbers: List of Z for elements considered inorganic (e.g., Pb, I)
        min_neighbors: Number of nearest neighbors to consider

    Returns:
        (cavity_radius, species_list, distances, coordination_number)
    """
    # Ionic radii (Shannon, approximate, in Å)
    IONIC_RADII = {
        "Pb": 1.19, "Sn": 0.69, "Ge": 0.53, "Ti": 0.61,
        "I": 2.20, "Br": 1.96, "Cl": 1.81, "F": 1.33,
        "O": 1.40, "S": 1.84, "Se": 1.98, "Te": 2.21,
        "Cs": 1.67, "Rb": 1.52, "K": 1.38, "Na": 1.02,
        "Bi": 1.03, "Sb": 0.76,
    }

    vc = virtual_site.coords
    distances = []
    species = []

    for site in structure:
        z = site.specie.Z
        if z < VIRTUAL_ELEMENT_OFFSET and z > 0:  # real inorganic atom
            dist = np.linalg.norm(site.coords - vc)
            distances.append(dist)
            species.append(str(site.specie.symbol))

    # Sort by distance
    sorted_pairs = sorted(zip(distances, species))[:min(50, len(distances))]
    sorted_distances, sorted_species = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # Cavity radius = distance to nearest neighbor - its ionic radius
    if sorted_distances:
        nearest_dist = sorted_distances[0]
        nearest_sp = str(sorted_species[0])
        ionic_r = IONIC_RADII.get(nearest_sp, 1.5)
        cavity_r = max(0.5, nearest_dist - ionic_r)
    else:
        cavity_r = 5.0

    # Coordination: count neighbors within 1.2x nearest distance
    if sorted_distances:
        cutoff = sorted_distances[0] * 1.2
        coord = sum(1 for d in sorted_distances if d < cutoff)
    else:
        coord = 6

    return cavity_r, list(sorted_species[:min_neighbors]), list(sorted_distances[:min_neighbors]), coord


def decode_virtual_to_molecule(
    virtual_z: int,
    virtual_library: VirtualElementLibrary,
    site_environment: LocalEnvironment,
    prefer_small: bool = True,
) -> Optional[str]:
    """
    Decode a virtual element to the best matching organic molecule SMILES.

    Args:
        virtual_z: Virtual atomic number
        virtual_library: Library with molecule mappings
        site_environment: Local crystal environment for cavity-aware selection
        prefer_small: If True, prefer molecules that fit snugly in cavity

    Returns:
        SMILES string of best matching molecule, or None if no candidates
    """
    candidates = virtual_library.get_smiles_from_virtual_z(virtual_z, n_candidates=10)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Score candidates based on local environment fit
    best_smi = None
    best_score = float("inf")

    for smi in candidates:
        desc = compute_descriptor_from_smiles(smi)
        # Score: absolute difference between molecular radius and cavity radius
        radius_diff = abs(desc.r_eff - site_environment.cavity_radius)
        # Penalize if too large (won't fit) more than too small
        if desc.r_eff > site_environment.cavity_radius:
            radius_diff *= 2.0

        # Bonus for charge matching with coordination
        charge_match = abs(desc.charge - (8 - site_environment.coordination_number) % 4)

        score = radius_diff + charge_match * 0.5

        if score < best_score:
            best_score = score
            best_smi = smi

    return best_smi


def generate_3d_conformer(smiles: str, n_conformers: int = 1) -> Optional["Chem.Mol"]:
    """
    Generate a 3D conformer for a molecule.

    Args:
        smiles: SMILES string
        n_conformers: Number of conformers to generate (return lowest energy)

    Returns:
        RDKit Mol with 3D coordinates, or None if generation fails
    """
    if not HAS_RDKIT:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)

    if success == 0:
        return None

    # Optimize all conformers with MMFF
    results = AllChem.MMFFOptimizeMoleculeConfs(mol)
    # Select lowest energy
    best_conf = min(results, key=lambda x: x[1] if x[0] == 0 else float("inf"))
    best_conf_id = best_conf[0] if isinstance(best_conf, tuple) else 0

    return mol


def estimate_orientation(
    mol: "Chem.Mol",
    local_env: LocalEnvironment,
    conf_id: int = -1,
) -> np.ndarray:
    """
    Estimate optimal orientation of a molecule in the crystal lattice.

    Strategy: orient the molecule's dipole moment toward the nearest electronegative
    atom (anion), and H-bond donors toward acceptors.

    Args:
        mol: RDKit Mol with 3D conformer
        local_env: Local crystal environment
        conf_id: Conformer ID to use

    Returns:
        3x3 rotation matrix
    """
    # For now: random rotation with seed based on environment
    # A production version would compute dipole moment and align it
    rng = np.random.RandomState(hash(str(local_env)) % (2**31 - 1))

    # Random rotation via random quaternion
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])

    # Quaternion to rotation matrix
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])

    return R


def place_molecule_at_site(
    mol_3d: "Chem.Mol",
    centroid: np.ndarray,
    orientation: np.ndarray,
    lattice: Lattice,
    conf_id: int = -1,
) -> List[PeriodicSite]:
    """
    Place a 3D molecule at a specific position in the crystal lattice.

    Args:
        mol_3d: RDKit Mol with 3D conformer
        centroid: Target position (Cartesian coordinates)
        orientation: 3x3 rotation matrix
        lattice: Crystal lattice
        conf_id: Conformer ID

    Returns:
        List of PeriodicSite for each atom in the molecule
    """
    conf = mol_3d.GetConformer(conf_id)
    sites = []

    for atom in mol_3d.GetAtoms():
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        # Center, rotate, translate
        mol_centroid = np.array([conf.GetAtomPosition(a.GetIdx()) for a in mol_3d.GetAtoms()]).mean(axis=0)
        pos_centered = pos - mol_centroid
        pos_rotated = orientation @ pos_centered
        pos_final = pos_rotated + centroid

        sites.append(PeriodicSite(
            species=Element(atom.GetSymbol()),
            coords=pos_final,
            lattice=lattice,
            coords_are_cartesian=True,
        ))

    return sites


def reconstruct_full_atom_structure(
    virtual_crystal: Structure,
    virtual_library: VirtualElementLibrary,
    relax: bool = False,
) -> Structure:
    """
    Reconstruct full-atom hybrid crystal from coarse-grained virtual element structure.

    Args:
        virtual_crystal: Structure with real + virtual element sites
        virtual_library: Library for decoding virtual elements
        relax: If True, attempt MLFF relaxation (requires mattersim)

    Returns:
        Full-atom Structure with organic molecules in place
    """
    # Inorganic atomic numbers for cavity estimation
    inorganic_z = [z for z in range(1, MAX_ATOMIC_NUM + 1) if z not in [1, 6, 7, 8]]

    full_atom_sites = []

    for site in virtual_crystal:
        z = site.specie.Z
        symbol = str(site.specie.symbol)

        # Detect virtual element by symbol pattern (X followed by digits)
        is_virtual_site = symbol.startswith("X") and len(symbol) > 1 and symbol[1:].isdigit()

        if not is_virtual_site:
            # Real inorganic atom, keep as-is
            full_atom_sites.append(site)
        else:
            # Virtual element - decode to organic molecule
            # Extract virtual Z from symbol
            virtual_z = int(symbol[1:]) if symbol[1:].isdigit() else VIRTUAL_ELEMENT_OFFSET
            # Estimate local environment
            cavity_r, neighbors, dists, coord = estimate_cavity_radius(
                site, virtual_crystal, inorganic_z
            )
            local_env = LocalEnvironment(
                position=site.coords,
                cavity_radius=cavity_r,
                nearest_inorganic_species=neighbors,
                nearest_distances=dists,
                coordination_number=coord,
            )

            # Decode
            smiles = decode_virtual_to_molecule(z, virtual_library, local_env)

            if smiles is None:
                # Fallback: use a placeholder atom
                print(f"[WARN] No molecule found for virtual element Z={virtual_z}, using placeholder")
                full_atom_sites.append(PeriodicSite(
                    species=Element("Xe"),  # placeholder
                    coords=site.coords,
                    lattice=virtual_crystal.lattice,
                    coords_are_cartesian=True,
                ))
                continue

            # Generate 3D conformer
            mol_3d = generate_3d_conformer(smiles)
            if mol_3d is None:
                print(f"[WARN] Failed to generate 3D conformer for {smiles}")
                continue

            # Orient
            orientation = estimate_orientation(mol_3d, local_env)

            # Place
            mol_sites = place_molecule_at_site(
                mol_3d, site.coords, orientation, virtual_crystal.lattice
            )
            full_atom_sites.extend(mol_sites)

    full_structure = Structure.from_sites(full_atom_sites)

    if relax:
        print("[INFO] MLFF relaxation requested but mattersim may not be available on Windows.")
        print("[INFO] Run relaxation manually with: mattersim relax structure.cif")

    return full_structure


def decode_generated_virtual_crystals(
    generated_structure_path: str,
    virtual_library: VirtualElementLibrary,
    output_path: str,
) -> Structure:
    """
    Convenience function to decode generated virtual element crystals to full-atom structures.

    Args:
        generated_structure_path: Path to generated .extxyz or .cif
        virtual_library: VirtualElementLibrary
        output_path: Output path for full-atom structure

    Returns:
        Full-atom hybrid structure
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.io import read as ase_read

    atoms = ase_read(generated_structure_path)
    structure = AseAtomsAdaptor.get_structure(atoms)

    full_structure = reconstruct_full_atom_structure(structure, virtual_library, relax=False)

    # Save to CIF
    full_structure.to(filename=output_path)
    print(f"[INFO] Saved full-atom structure to {output_path}")

    return full_structure
