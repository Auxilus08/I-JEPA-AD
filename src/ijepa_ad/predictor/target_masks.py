"""Training target masks that mirror the masks used at inference.

Inference (``ijepa_ad.inference.MaskedLatentScorer``) hides one contiguous coarse
window at a time (the grid-aligned windows of ``CoarseMasker``: 4×4 tokens, with
narrower windows on the 14-token grid edge) and then single tokens. This sampler
draws the same two kinds of targets so the predictor is trained on the
missing-region geometry it is scored on.

Only the *selection* of target positions lives here. Target latents are replaced by
the predictor's mask token inside ``LatentTransformerPredictor.forward``, so no mask
choice can leak a ground-truth target latent into the context.
"""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor

from ijepa_ad.masking import CoarseMasker, PatchGrid

_PATCH_SIZE = 16  # only used to build a PatchGrid of the right token-grid size
_SAMPLE_SEED_DOMAIN = "ijepa-ad/validation-mask/v1"


def sample_mask_seed(base_seed: int, sample_id: str, epoch: int | None = None) -> int:
    """Derive a stable 63-bit generator seed for one sample's validation mask.

    The seed is the first 8 bytes (big-endian, top bit cleared) of
    ``SHA-256("<domain>|<base_seed>|<epoch or ->|<sample_id>")``. SHA-256 is
    process- and platform-independent, unlike Python's salted ``hash()``. The seed
    depends only on ``base_seed``, the sample id and, if given, the epoch, so a
    sample's mask does not depend on batch size, batch order or batch neighbours.
    ``epoch=None`` yields one fixed mask per sample for every epoch.
    """
    epoch_part = "-" if epoch is None else str(epoch)
    text = f"{_SAMPLE_SEED_DOMAIN}|{base_seed}|{epoch_part}|{sample_id}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_inference_aligned_target_mask(
    batch_size: int,
    num_tokens: int,
    generator: torch.Generator,
    *,
    block_size: int = 4,
    block_probability: float = 0.5,
    single_token_count: int = 8,
) -> Tensor:
    """Sample a boolean ``[batch_size, num_tokens]`` target mask, one kind per row.

    With probability ``block_probability`` a row targets exactly one of the coarse
    inference windows (a contiguous block). Otherwise it targets up to
    ``single_token_count`` isolated tokens: no two are 4-neighbours, so each target
    keeps visible neighbours as in a 1×1 inference mask. Several singles per row
    give more loss terms per forward pass than one. Every row keeps at least one
    target and at least one context token.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    side = math.isqrt(num_tokens)
    if side * side != num_tokens:
        raise ValueError(f"num_tokens must be a square token grid, got {num_tokens}")
    if not 1 <= block_size < side:
        raise ValueError(
            f"block_size must be in [1, {side - 1}] so context remains, "
            f"got {block_size}"
        )
    if not 0.0 <= block_probability <= 1.0:
        raise ValueError("block_probability must be in [0, 1]")
    if single_token_count < 1:
        raise ValueError("single_token_count must be at least 1")

    grid = PatchGrid(image_size=side * _PATCH_SIZE, patch_size=_PATCH_SIZE)
    windows = CoarseMasker(grid, block_size).masks()
    use_block = torch.rand(batch_size, generator=generator) < block_probability
    window_index = torch.randint(len(windows), (batch_size,), generator=generator)

    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    for row in range(batch_size):
        if bool(use_block[row]):
            mask[row] = windows[int(window_index[row])]
        else:
            order = torch.randperm(num_tokens, generator=generator).tolist()
            mask[row, _isolated_tokens(order, side, single_token_count)] = True
    return mask


def _isolated_tokens(order: list[int], side: int, count: int) -> list[int]:
    chosen: list[int] = []
    taken: set[int] = set()
    for token in order:
        row, column = divmod(token, side)
        neighbours = (
            token - side if row else -1,
            token + side if row + 1 < side else -1,
            token - 1 if column else -1,
            token + 1 if column + 1 < side else -1,
        )
        if any(neighbour in taken for neighbour in neighbours):
            continue
        chosen.append(token)
        taken.add(token)
        if len(chosen) == count:
            break
    return chosen
