# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Extended atom type embedding for MatterGen that supports both real elements (Z <= MAX_ATOMIC_NUM)
and virtual elements (Z >= VIRTUAL_ELEMENT_OFFSET).

Replaces the original AtomEmbedding from mattergen.common.gemnet.layers.embedding_block.
"""

import numpy as np
import torch

from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM,
    VIRTUAL_ELEMENT_OFFSET,
    N_VIRTUAL_ELEMENTS,
    EXTENDED_VOCAB_SIZE,
    is_virtual_element,
    get_virtual_element_index,
)
from mattergen.molecule_mapping.library import VirtualElementLibrary, build_quick_perovskite_library


class ExtendedAtomEmbedding(torch.nn.Module):
    """
    Extended atom embedding layer supporting real elements and virtual elements.

    Real elements (Z <= MAX_ATOMIC_NUM):
        Use learned embeddings (initialized from MatterGen pretrained weights)

    Virtual elements (Z >= VIRTUAL_ELEMENT_OFFSET):
        Use embeddings initialized from molecular descriptors/cluster centers
        Optionally frozen to preserve molecular semantics

    Mask token (Z = VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS):
        Separate learnable embedding for D3PM mask diffusion

    Parameters
    ----------
        emb_size: int
            Atom embedding size
        with_mask_type: bool
            Whether to include mask type (for D3PM diffusion)
        virtual_element_library: VirtualElementLibrary, optional
            Library providing virtual element embeddings. If None, builds a
            default perovskite cation library automatically.
        freeze_mol_embeddings: bool
            If True, freeze virtual element embeddings during training
    """

    def __init__(
        self,
        emb_size: int,
        virtual_element_library: VirtualElementLibrary | None = None,
        with_mask_type: bool = False,
        freeze_mol_embeddings: bool = True,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.with_mask_type = with_mask_type

        # Auto-build default library if not provided
        if virtual_element_library is None:
            virtual_element_library = build_quick_perovskite_library()
        self._virtual_library = virtual_element_library

        # Real element embeddings: Z in [1, MAX_ATOMIC_NUM], indexed as Z-1
        real_vocab_size = MAX_ATOMIC_NUM + int(with_mask_type)
        self.real_embeddings = torch.nn.Embedding(real_vocab_size, emb_size)
        torch.nn.init.uniform_(self.real_embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

        # Virtual element embeddings: fixed capacity for N_VIRTUAL_ELEMENTS
        self.n_virtual = N_VIRTUAL_ELEMENTS
        initial_weights = torch.zeros(N_VIRTUAL_ELEMENTS, emb_size)
        for i in range(N_VIRTUAL_ELEMENTS):
            vz = VIRTUAL_ELEMENT_OFFSET + i
            initial_weights[i] = virtual_element_library.get_embedding(vz, emb_size)

        self.virtual_embeddings = torch.nn.Embedding.from_pretrained(
            initial_weights, freeze=freeze_mol_embeddings
        )

        # Mask token embedding (Z = VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS)
        # This corresponds to the absorbing state in D3PM mask diffusion
        self.mask_embedding = torch.nn.Parameter(
            torch.empty(1, emb_size)
        )
        torch.nn.init.uniform_(self.mask_embedding, a=-np.sqrt(3), b=np.sqrt(3))

    @property
    def full_vocab_size(self) -> int:
        """Total vocabulary size for D3PM (real + virtual + mask)."""
        return EXTENDED_VOCAB_SIZE

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z: torch.Tensor, shape=(nAtoms,) - atomic numbers
               Real: 1 to MAX_ATOMIC_NUM
               Virtual: VIRTUAL_ELEMENT_OFFSET to VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS - 1
               Mask: VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS

        Returns:
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        mask_z = VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS
        is_mask = Z == mask_z
        is_virtual = (Z >= VIRTUAL_ELEMENT_OFFSET) & ~is_mask
        is_real = (Z > 0) & (Z <= MAX_ATOMIC_NUM)

        # Initialize output
        h = torch.zeros(len(Z), self.emb_size, device=Z.device, dtype=torch.float32)

        # Real elements: Z-1 indexing (as in original AtomEmbedding)
        if is_real.any():
            real_Z = Z[is_real].long()
            h[is_real] = self.real_embeddings(real_Z - 1)

        # Virtual elements: offset indexing
        if is_virtual.any():
            virtual_indices = get_virtual_element_index(Z[is_virtual].long())
            h[is_virtual] = self.virtual_embeddings(virtual_indices)

        # Mask token
        if is_mask.any():
            h[is_mask] = self.mask_embedding.expand(is_mask.sum(), -1)

        return h

    def load_real_weights_from_pretrained(self, pretrained_embeddings: torch.nn.Embedding):
        """
        Load real element embeddings from a pretrained MatterGen model.

        Args:
            pretrained_embeddings: The original AtomEmbedding.embeddings from a pretrained model
        """
        with torch.no_grad():
            self.real_embeddings.weight.data[:pretrained_embeddings.weight.shape[0]] = \
                pretrained_embeddings.weight.data.clone()
