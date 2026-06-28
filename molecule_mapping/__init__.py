# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Molecule Mapping module for organic-inorganic hybrid crystal generation.

Exports:
- VirtualElementLibrary: Build and manage virtual element mappings
- ExtendedAtomEmbedding: Extended atom embedding for real + virtual elements
- convert_hybrid_to_virtual_crystal: Data conversion utility
- reconstruct_full_atom_structure: Full-atom reconstruction decoder
"""

from mattergen.molecule_mapping.virtual_elements import (
    N_VIRTUAL_ELEMENTS,
    VIRTUAL_ELEMENT_OFFSET,
    EXTENDED_ATOMIC_NUM,
    EXTENDED_VOCAB_SIZE,
    VIRTUAL_VOLUME_FACTOR,
    MOLECULE_ROLES,
    MOLECULE_ROLE_A_SITE,
    MOLECULE_ROLE_SPACER,
    MOLECULE_ROLE_LINKER,
    is_virtual_element,
    is_real_element,
    get_virtual_element_index,
    get_virtual_element_z,
    get_virtual_symbol,
)

from mattergen.molecule_mapping.descriptors import (
    MolecularDescriptor,
    compute_descriptor_from_smiles,
    HAS_RDKIT,
)

from mattergen.molecule_mapping.library import (
    VirtualElementLibrary,
    VirtualElementInfo,
    build_virtual_element_library,
    build_identity_library,
    build_quick_perovskite_library,
    KNOWN_PEROVSKITE_CATIONS,
)

from mattergen.molecule_mapping.data_convert import (
    convert_hybrid_to_virtual_crystal,
    convert_virtual_to_real_atomic_numbers,
    get_virtual_mask,
    identify_organic_fragments,
)

from mattergen.molecule_mapping.extended_embedding import (
    ExtendedAtomEmbedding,
)

from mattergen.molecule_mapping.decoder import (
    LocalEnvironment,
    decode_virtual_to_molecule,
    generate_3d_conformer,
    place_molecule_at_site,
    reconstruct_full_atom_structure,
    decode_generated_virtual_crystals,
)

from mattergen.molecule_mapping.ion_classes import (
    IonClass,
    ION_CLASSES,
    ION_CLASS_BY_Z,
    ION_CLASS_BY_INDEX,
    ION_CLASS_BY_NAME,
    CATION_TO_CLASS,
    get_ion_class_for_cation,
    get_ion_class_by_z,
    CLASS_Z_OFFSET,
)

__all__ = [
    # Virtual elements
    "N_VIRTUAL_ELEMENTS",
    "VIRTUAL_ELEMENT_OFFSET",
    "EXTENDED_ATOMIC_NUM",
    "EXTENDED_VOCAB_SIZE",
    "VIRTUAL_VOLUME_FACTOR",
    "MOLECULE_ROLES",
    "MOLECULE_ROLE_A_SITE",
    "MOLECULE_ROLE_SPACER",
    "MOLECULE_ROLE_LINKER",
    "is_virtual_element",
    "is_real_element",
    "get_virtual_element_index",
    "get_virtual_element_z",
    "get_virtual_symbol",
    "mask_unused_virtual_elements",
    # Descriptors
    "MolecularDescriptor",
    "compute_descriptor_from_smiles",
    "HAS_RDKIT",
    # Library
    "VirtualElementLibrary",
    "VirtualElementInfo",
    "build_virtual_element_library",
    "build_identity_library",
    "build_quick_perovskite_library",
    "KNOWN_PEROVSKITE_CATIONS",
    # Data conversion
    "convert_hybrid_to_virtual_crystal",
    "convert_virtual_to_real_atomic_numbers",
    "get_virtual_mask",
    "identify_organic_fragments",
    # Extended embedding
    "ExtendedAtomEmbedding",
    # Decoder
    "LocalEnvironment",
    "decode_virtual_to_molecule",
    "generate_3d_conformer",
    "place_molecule_at_site",
    "reconstruct_full_atom_structure",
    "decode_generated_virtual_crystals",
    # Ion classes (compressed taxonomy)
    "IonClass",
    "ION_CLASSES",
    "ION_CLASS_BY_Z",
    "ION_CLASS_BY_INDEX",
    "ION_CLASS_BY_NAME",
    "CATION_TO_CLASS",
    "get_ion_class_for_cation",
    "get_ion_class_by_z",
    "CLASS_Z_OFFSET",
]
