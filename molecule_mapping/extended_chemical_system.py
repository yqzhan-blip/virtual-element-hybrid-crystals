# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Extended chemical system multi-hot embedding that supports virtual elements.

Extends the original ChemicalSystemMultiHotEmbedding to handle virtual elements (Z >= VIRTUAL_ELEMENT_OFFSET)
in the chemical system conditioning vector, enabling generation conditioned on both
real elements AND virtual elements (organic molecules).
"""

from typing import Sequence

import torch

from pymatgen.core import Element

from mattergen.molecule_mapping.virtual_elements import (
    MAX_ATOMIC_NUM,
    VIRTUAL_ELEMENT_OFFSET,
    N_VIRTUAL_ELEMENTS,
    EXTENDED_VOCAB_SIZE,
    is_virtual_element,
    is_real_element,
    get_virtual_element_index,
)


def _get_atomic_number(symbol: str) -> int:
    """Local version to avoid importing mattergen.common.utils.data_utils (requires torch_geometric)."""
    return Element(symbol).Z


class ExtendedChemicalSystemMultiHotEmbedding(torch.nn.Module):
    """
    Extended chemical system embedding supporting virtual elements.

    Chemical system is specified as a list of element/virtual-element symbols:
    e.g., ["Pb", "I", "X119"] means the system contains Pb, I, and virtual element 119 (MA+).

    The multi-hot vector dimension is EXTENDED_VOCAB_SIZE = MAX_ATOMIC_NUM + N_VIRTUAL_ELEMENTS + 1,
    where the last position is reserved for the mask token.

    Parameters
    ----------
        hidden_dim: int
            Output embedding dimension
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Extended vocabulary: real elements + virtual elements
        self.embedding = torch.nn.Linear(in_features=EXTENDED_VOCAB_SIZE, out_features=hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _sequence_to_multi_hot(x: Sequence[str], device: torch.device) -> torch.Tensor:
        """
        Convert a sequence of element/virtual-element symbols to a multi-hot vector.

        Examples:
            ["Pb", "I", "X119"] -> multi-hot of length EXTENDED_VOCAB_SIZE

        Returns:
            torch.Tensor, shape=(1, EXTENDED_VOCAB_SIZE)
        """
        multi_hot = torch.zeros(EXTENDED_VOCAB_SIZE, device=device)

        for symbol in x:
            if symbol.startswith("X") and len(symbol) > 1:
                # Virtual element: X120, X150, etc.
                try:
                    vz = int(symbol[1:])
                    if is_virtual_element(vz):
                        idx = get_virtual_element_index(vz)
                        # Virtual elements occupy indices [MAX_ATOMIC_NUM + 1, EXTENDED_VOCAB_SIZE)
                        multi_hot[MAX_ATOMIC_NUM + 1 + idx] = 1.0
                    else:
                        print(f"[WARN] Unknown virtual element: {symbol}")
                except ValueError:
                    print(f"[WARN] Could not parse virtual element: {symbol}")
            else:
                # Real element
                try:
                    z = _get_atomic_number(symbol=_system_symbol_to_element(symbol))
                    if 0 < z <= MAX_ATOMIC_NUM:
                        multi_hot[z] = 1.0
                except Exception:
                    print(f"[WARN] Unknown element: {symbol}")

        return multi_hot.reshape(1, -1)

    @staticmethod
    def sequences_to_multi_hot(x: list[list[str]], device: torch.device) -> torch.Tensor:
        """
        Convert a batch of chemical system sequences to multi-hot tensors.

        Returns:
            torch.Tensor, shape=(n_structures_in_batch, EXTENDED_VOCAB_SIZE)
        """
        return torch.cat(
            [ExtendedChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(_x, device=device) for _x in x],
            dim=0,
        )

    @staticmethod
    def convert_to_list_of_str(x: list[str] | list[list[str]]) -> list[list[str]]:
        """
        Normalize chemical system input to list of list of str format.

        Handles:
        - "Bi-I" -> ["Bi", "I"]
        - "Pb-I-X119" -> ["Pb", "I", "X119"]
        - ["Pb", "I", "X119"] -> [["Pb", "I", "X119"]]
        """
        if isinstance(x[0], str):
            # list of string formulas: ["Bi-I", "Pb-I-X119", ...]
            return [_x.split("-") for _x in x if isinstance(_x, str)]
        return x  # type: ignore

    def forward(self, x: list[str] | list[list[str]]) -> torch.Tensor:
        """
        Args:
            x: Chemical system specification
               - list[str]: ["Pb-I-X119", "Pb-I-X120", ...]
               - list[list[str]]: [["Pb", "I", "X119"], ["Pb", "I", "X120"], ...]

        Returns:
            torch.Tensor, shape=(n_structures_in_batch, hidden_dim)
        """
        x = self.convert_to_list_of_str(x=x)
        multi_hot = self.sequences_to_multi_hot(x=x, device=self.device)
        return self.embedding(multi_hot)


def _system_symbol_to_element(symbol: str) -> str:
    """
    Normalize element symbol for pymatgen compatibility.
    Handles cases like "He" -> "He", "na" -> "Na".

    Args:
        symbol: Element symbol (case-insensitive)

    Returns:
        Properly capitalized element symbol
    """
    if len(symbol) == 1:
        return symbol.upper()
    elif len(symbol) == 2:
        return symbol[0].upper() + symbol[1].lower()
    else:
        return symbol[0].upper() + symbol[1:].lower()
