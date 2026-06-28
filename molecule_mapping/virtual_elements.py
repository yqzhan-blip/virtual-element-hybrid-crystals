# Copyright (c) Microsoft Corporation. (Extended for organic-inorganic hybrid crystals)
# Licensed under the MIT License.

"""
Virtual element constants for organic-inorganic hybrid crystal generation.
Extends MatterGen's atomic number space with "virtual elements" (Z >= VIRTUAL_ELEMENT_OFFSET).
Each virtual element represents a class of organic ions with similar crystal-chemical roles.

Architecture (with gap masking):
  Z = 1..118       → real elements              → D3PM indices 0..117
  Z = 119..200     → GAP (always masked)        → D3PM indices 118..199
  Z = 201..200+N   → virtual elements (active)  → D3PM indices 200..200+N-1
  Z = 200+N+1..400 → unused virtual (masked)   → D3PM indices 200+N..399
  Z = 401 (mask)   → D3PM mask token           → D3PM index 400
  VIRTUAL_ELEMENT_OFFSET = 201
"""

# Atomic number of the last real element (Oganesson, Og)
MAX_ATOMIC_NUM = 118

# Virtual elements start at 201 — well beyond the real periodic table.
# Z=119-200 is a strategic gap (always masked at sampling time) to visually
# separate real and virtual elements in the extended periodic table.
VIRTUAL_ELEMENT_OFFSET = 201

# Total virtual element slots in the embedding layer (matches original training).
# Active slots = number of ion classes (12 currently, read from ION_CLASSES).
# Remaining slots are masked at sampling but kept for future expansion.
N_VIRTUAL_ELEMENTS = 200

# Extended vocabulary size for D3PM atom type diffusion.
# Schema: Z=1..118 (real) | Z=119..200 (gap) | Z=201..400 (virtual) | +1 (mask)
EXTENDED_ATOMIC_NUM = VIRTUAL_ELEMENT_OFFSET + N_VIRTUAL_ELEMENTS - 1  # = 400
EXTENDED_VOCAB_SIZE = EXTENDED_ATOMIC_NUM + 1  # = 401

# Volume ratio
VIRTUAL_VOLUME_FACTOR = 27.0

# Hydrogen bond energy scale (kJ/mol)
HBOND_ENERGY_SCALE = 20.0

# Organic ion role categories
MOLECULE_ROLE_A_SITE = "A-site cation"
MOLECULE_ROLE_SPACER = "spacer cation"
MOLECULE_ROLE_LINKER = "linker"

MOLECULE_ROLES = [MOLECULE_ROLE_A_SITE, MOLECULE_ROLE_SPACER, MOLECULE_ROLE_LINKER]


def is_virtual_element(z: int) -> bool:
    """Check if an atomic number corresponds to a virtual element."""
    return z >= VIRTUAL_ELEMENT_OFFSET


def get_virtual_element_index(z) -> "int | Tensor":
    """Convert virtual element atomic number to zero-based index."""
    return z - VIRTUAL_ELEMENT_OFFSET


def get_virtual_element_z(index: int) -> int:
    """Convert zero-based virtual element index to atomic number."""
    return index + VIRTUAL_ELEMENT_OFFSET


def get_virtual_symbol(z: int) -> str:
    """Get display symbol for a virtual element, e.g., X201, X202, ..."""
    return f"X{z}"


def is_real_element(z: int) -> bool:
    """Check if an atomic number corresponds to a real element."""
    return 0 < z <= MAX_ATOMIC_NUM


###############################################################################
# D3PM Element Masking — Exclude gap + unused virtual during sampling
###############################################################################


def _get_active_virtual_count() -> int:
    """Return number of active virtual element slots (lazy import from ION_CLASSES)."""
    try:
        from mattergen.molecule_mapping.ion_classes import ION_CLASSES
        return len(ION_CLASSES)
    except ImportError:
        return 200  # fallback: all virtual slots


def _build_allowed_z_ranges() -> list[tuple[int, int]]:
    """Build allowed Z ranges: real elements + active virtual classes only."""
    n_active = _get_active_virtual_count()
    virtual_max = VIRTUAL_ELEMENT_OFFSET + n_active - 1  # inclusive, e.g. 201+12-1=212
    return [
        (1, MAX_ATOMIC_NUM),                        # real: Z=1..118
        (VIRTUAL_ELEMENT_OFFSET, virtual_max),       # virtual: Z=201..212
    ]


# Dynamically computed allowed ranges
ALLOWED_Z_RANGES = _build_allowed_z_ranges()


def get_allowed_atomic_numbers() -> list[int]:
    """Return flat list of all allowed atomic numbers (real + active virtual)."""
    allowed = []
    for lo, hi in ALLOWED_Z_RANGES:
        allowed.extend(range(lo, hi + 1))
    return allowed


def mask_unused_virtual_elements(
    logits,        # torch.Tensor, shape (N, vocab_size)
    x=None,        # unused, for interface compatibility
    batch_idx=None, # unused
    predictions_are_zero_based: bool = True,
):
    """Mask D3PM logits to exclude:
      - Gap region Z=119-200 (always masked)
      - Unused virtual slots Z > 200+N (where N = active ion classes)

    Only Z=1..118 (real) + Z=201..200+N (active virtual) + MASK token are kept.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size), vocab_size=401
        predictions_are_zero_based: If True, D3PM index i ↔ Z=i+1

    Returns:
        Masked logits with disallowed indices set to -inf
    """
    import torch

    vocab_size = logits.shape[1]
    device = logits.device

    keep_mask = torch.zeros(vocab_size, device=device)

    for lo, hi in ALLOWED_Z_RANGES:
        if predictions_are_zero_based:
            start_idx = lo - 1   # D3PM index = Z - 1
            end_idx = hi          # exclusive
        else:
            start_idx = lo
            end_idx = hi + 1

        if start_idx < vocab_size and end_idx <= vocab_size:
            keep_mask[start_idx:end_idx] = 1.0

    # MASK token (last index)
    if vocab_size > 0:
        keep_mask[-1] = 1.0

    masked_logits = logits + torch.log(keep_mask[None, :])
    return masked_logits
