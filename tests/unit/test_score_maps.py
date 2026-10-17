"""Tests for patch-score upsampling and the display-only heatmap normalization."""

import torch
import torch.nn.functional as F

from ijepa_ad.masking import patch_scores_to_heatmap, upsample_patch_scores


def test_upsample_matches_bilinear_align_corners_false_on_raw_scores() -> None:
    scores = torch.rand(196) * 3
    expected = F.interpolate(
        scores.reshape(1, 1, 14, 14),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    assert torch.equal(upsample_patch_scores(scores), expected)


def test_upsample_supports_rectangular_ground_truth_sizes_and_batches() -> None:
    maps = upsample_patch_scores(torch.rand(3, 196), output_size=(900, 1000))

    assert maps.shape == (3, 1, 900, 1000)
    assert maps.dtype == torch.float32


def test_raw_upsampling_preserves_score_scale_and_is_not_nearest() -> None:
    scores = torch.arange(196, dtype=torch.float32)
    raw = upsample_patch_scores(scores)

    assert float(raw.min()) >= float(scores.min())
    assert float(raw.max()) <= float(scores.max())
    assert raw.unique().numel() > 196  # interpolated, not blocky
    constant = upsample_patch_scores(torch.full((196,), 0.37))
    assert torch.allclose(constant, torch.full_like(constant, 0.37))


def test_upsampling_does_not_rescale_each_image_independently() -> None:
    base = torch.rand(196)
    both = upsample_patch_scores(torch.stack([base, base * 5]))

    assert torch.allclose(both[1], both[0] * 5, rtol=1e-5)  # magnitude retained


def test_heatmap_is_min_max_of_the_same_interpolation() -> None:
    scores = torch.rand(2, 196)
    raw = upsample_patch_scores(scores)
    heatmap = patch_scores_to_heatmap(scores)

    assert heatmap.shape == (2, 1, 224, 224)
    assert torch.equal(patch_scores_to_heatmap(scores, normalize=False), raw)
    for index in range(2):
        low, high = raw[index].min(), raw[index].max()
        assert torch.allclose(
            heatmap[index], (raw[index] - low) / (high - low), atol=1e-6
        )
        assert torch.equal(heatmap[index].argmax(), raw[index].argmax())


def test_flat_score_map_normalizes_to_zeros_without_nans() -> None:
    heatmap = patch_scores_to_heatmap(torch.full((196,), 0.5))

    assert torch.isfinite(heatmap).all() and float(heatmap.max()) == 0.0


def test_token_to_pixel_mapping_is_row_major() -> None:
    scores = torch.zeros(196)
    scores[1 * 14 + 3] = 1.0  # row 1, column 3 -> pixel rows 16..31, cols 48..63

    row, column = divmod(int(upsample_patch_scores(scores)[0, 0].argmax()), 224)

    assert 16 <= row < 32 and 48 <= column < 64
