# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Data conversion utilities for organic-inorganic hybrid crystal structures.

Converts between:
- Full-atom hybrid structures (pymatgen Structure with organic molecules)
- Coarse-grained virtual-element structures (virtual elements as single sites)
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Element, Lattice, PeriodicSite, Species, Structure

from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM,
    VIRTUAL_ELEMENT_OFFSET,
    is_virtual_element,
    is_real_element,
    get_virtual_element_z,
    get_virtual_symbol,
)
from mattergen.molecule_mapping.descriptors import (
    MolecularDescriptor,
    compute_descriptor_from_smiles,
)
from mattergen.molecule_mapping.library import VirtualElementLibrary


def identify_organic_fragments(
    structure: Structure,
    organic_elements: Optional[List[str]] = None,
) -> Tuple[List[List[PeriodicSite]], List[PeriodicSite]]:
    """
    Identify organic and inorganic sites in a hybrid crystal structure.

    Organic fragments are identified as connected components of "organic" elements
    (C, N, O, H, etc.) that are bonded together.

    Args:
        structure: pymatgen Structure with organic molecules
        organic_elements: List of element symbols to consider "organic"
                         Default: ["C", "N", "O", "H", "S", "P", "F", "Cl", "Br", "I"]

    Returns:
        (organic_fragments, inorganic_sites)
        - organic_fragments: List of lists of PeriodicSite (each inner list is one molecule)
        - inorganic_sites: List of individual inorganic PeriodicSites
    """
    if organic_elements is None:
        organic_elements = ["C", "N", "O", "H", "S", "P", "F", "Cl", "Br", "I"]

    organics_set = set(organic_elements)

    # Classify sites
    organic_sites = []
    inorganic_sites = []
    for site in structure:
        symbol = str(site.specie.symbol)
        if symbol in organics_set:
            organic_sites.append(site)
        else:
            inorganic_sites.append(site)

    # Simple heuristic: group organic sites by proximity
    # In a real implementation, would use bond detection
    # For now, group by spatial proximity (distance < 2.0 Å = bond)
    fragments = group_by_bonding(organic_sites, structure.lattice, bond_cutoff=2.0)

    return fragments, inorganic_sites


def group_by_bonding(
    sites: List[PeriodicSite],
    lattice: Lattice,
    bond_cutoff: float = 2.0,
) -> List[List[PeriodicSite]]:
    """
    Group sites into connected fragments based on bonding distance.

    Args:
        sites: List of sites to group
        lattice: Crystal lattice
        bond_cutoff: Maximum distance for two atoms to be considered bonded

    Returns:
        List of fragments (each fragment is a list of connected sites)
    """
    if not sites:
        return []

    n = len(sites)
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    coords = np.array([s.coords for s in sites])

    for i in range(n):
        for j in range(i + 1, n):
            # Compute minimum image distance
            diff = coords[i] - coords[j]
            # Fractional difference
            frac_diff = lattice.get_fractional_coords(diff)
            # Wrap to [-0.5, 0.5]
            frac_diff = frac_diff - np.round(frac_diff)
            # Convert back to Cartesian
            cart_diff = lattice.get_cartesian_coords(frac_diff)
            dist = np.linalg.norm(cart_diff)
            if dist < bond_cutoff:
                adj[i, j] = adj[j, i] = True

    # Find connected components via DFS
    visited = np.zeros(n, dtype=bool)
    fragments = []

    for i in range(n):
        if not visited[i]:
            # DFS
            stack = [i]
            component = []
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor in range(n):
                    if adj[node, neighbor] and not visited[neighbor]:
                        stack.append(neighbor)
            fragments.append([sites[idx] for idx in component])

    return fragments


def compute_centroid(sites: List[PeriodicSite]) -> np.ndarray:
    """Compute center of mass (unweighted centroid) of a list of sites."""
    coords = np.array([s.coords for s in sites])
    return coords.mean(axis=0)


def convert_hybrid_to_virtual_crystal(
    hybrid_structure: Structure,
    virtual_library: VirtualElementLibrary,
    organic_smiles_map: Optional[dict] = None,
) -> Structure:
    """
    Convert a full-atom hybrid crystal to a coarse-grained virtual-element crystal.

    Steps:
    1. Identify organic fragments (connected C/N/O/H atoms)
    2. Compute centroid for each fragment
    3. Map each fragment to nearest virtual element
    4. Build new structure: inorganic sites + virtual element nodes

    Args:
        hybrid_structure: Full-atom structure with organic molecules
        virtual_library: VirtualElementLibrary for mapping
        organic_smiles_map: Optional dict mapping fragment index to SMILES
                           (if None, we use charge=+1 as default)

    Returns:
        Coarse-grained Structure with real and virtual element sites
    """
    fragments, inorganic_sites = identify_organic_fragments(hybrid_structure)

    # Create virtual sites
    virtual_sites = []
    for i, fragment in enumerate(fragments):
        centroid = compute_centroid(fragment)

        # Get SMILES if available
        if organic_smiles_map and i in organic_smiles_map:
            smi_info = organic_smiles_map[i]
            if isinstance(smi_info, str):
                smi = smi_info
                charge = 1
                role = "A-site cation"
            else:
                smi = smi_info["smiles"]
                charge = smi_info.get("charge", 1)
                role = smi_info.get("role", "A-site cation")
            vz = virtual_library.get_virtual_z(smi, charge=charge, role=role)
        else:
            # Default: charge +1, A-site cation
            # Estimate from fragment composition
            vz = VIRTUAL_ELEMENT_OFFSET  # fallback to first virtual element

        virtual_site = PeriodicSite(
            species=Species.from_string(get_virtual_symbol(vz)),
            coords=centroid,
            lattice=hybrid_structure.lattice,
            coords_are_cartesian=True,
        )
        virtual_sites.append(virtual_site)

    # Build new structure
    all_sites = inorganic_sites + virtual_sites
    return Structure.from_sites(all_sites)


def convert_virtual_to_real_atomic_numbers(
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """
    Convert virtual element atomic numbers back to a representative real element
    for compatibility with radii-based operations in MatterGen.

    Args:
        atomic_numbers: shape (N_atoms,) with values in [0, EXTENDED_ATOMIC_NUM]

    Returns:
        Same shape, with virtual elements mapped to their most similar real element
        (Bismuth, Z=83, as a heavy representative for most organic cations)
    """
    # Map virtual elements to Bi (83) as a heavy placeholder
    mask = atomic_numbers >= VIRTUAL_ELEMENT_OFFSET
    result = atomic_numbers.clone()
    result[mask] = 83  # Bi
    return result


def get_virtual_mask(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Get boolean mask for virtual element positions."""
    return atomic_numbers >= VIRTUAL_ELEMENT_OFFSET
