"""Tests for the opt-in scoring variants (discrepancies, normalisers, smoothing)."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from ijepa_ad.inference import InferenceMode, MaskedLatentScorer
from ijepa_ad.masking.score_maps import smooth_score_maps
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)
from ijepa_ad.scoring import (
    ChannelStandardisedCosine,
    ChannelStats,
    PositionNormaliser,
    cosine_discrepancy,
    fit_channel_stats,
    fit_mahalanobis_residual,
    layer_norm_cosine_discrepancy,
)

DIM = 12


def _predictor() -> LatentTransformerPredictor:
    torch.manual_seed(0)
    config = PredictorConfig(
        embedding_dim=DIM,
        num_tokens=196,
        hidden_dim=16,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    predictor = LatentTransformerPredictor(config)
    predictor.eval()
    return predictor


def _latents(seed: int = 1) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 196, DIM, generator=generator)


def _original_score_windows(scorer: MaskedLatentScorer, latents: Tensor, windows):  # type: ignore[no-untyped-def]
    """Verbatim copy of ``_score_windows`` before the discrepancy option existed."""
    scores = latents.new_zeros(scorer.grid.tokens)
    if not windows:
        return scores
    masks = torch.zeros((len(windows), scorer.grid.tokens), dtype=torch.bool)
    for row, window in enumerate(windows):
        masks[row, list(window.token_indices)] = True
    with torch.inference_mode():
        for start in range(0, len(windows), scorer.mask_batch_size):
            stop = min(start + scorer.mask_batch_size, len(windows))
            batch_masks = masks[start:stop]
            batch_latents = latents.expand(stop - start, -1, -1)
            predicted = scorer.predictor(batch_latents, batch_masks)
            discrepancies = (
                1 - F.cosine_similarity(predicted, batch_latents, dim=-1, eps=1e-8)
            ).clamp_(0, 2)
            for row, window in enumerate(windows[start:stop]):
                indices = list(window.token_indices)
                scores[indices] = discrepancies[row, indices]
    return scores


@pytest.mark.parametrize("coarse_size", [4, 2])  # type: ignore[misc]
def test_default_scorer_is_bit_identical_to_the_original_algorithm(
    coarse_size: int,
) -> None:
    scorer = MaskedLatentScorer(_predictor(), coarse_size=coarse_size)
    latents = _latents()

    coarse = scorer._score_windows(latents, scorer.coarse.windows())
    dense = scorer._score_windows(latents, scorer.fine.candidates())

    assert torch.equal(
        coarse, _original_score_windows(scorer, latents, scorer.coarse.windows())
    )
    assert torch.equal(
        dense, _original_score_windows(scorer, latents, scorer.fine.candidates())
    )


@pytest.mark.parametrize("mode", ["adaptive", "dense", "fixed"])  # type: ignore[misc]
def test_explicit_cosine_discrepancy_matches_the_default(mode: InferenceMode) -> None:
    latents = _latents()
    default = MaskedLatentScorer(_predictor(), threshold=0.0).score(latents, mode=mode)
    explicit = MaskedLatentScorer(
        _predictor(), threshold=0.0, discrepancy=cosine_discrepancy
    ).score(latents, mode=mode)

    assert torch.equal(default.patch_scores, explicit.patch_scores)
    assert default.evaluated_regions == explicit.evaluated_regions


def test_custom_discrepancy_changes_scores_but_not_the_windows_evaluated() -> None:
    latents = _latents()
    huge = 1e6

    def constant(predicted: Tensor, actual: Tensor) -> Tensor:
        return torch.full(predicted.shape[:2], huge)

    # A mid threshold fires on some but not all regions of a random predictor.
    default = MaskedLatentScorer(_predictor(), threshold=1.0).score(latents)
    custom = MaskedLatentScorer(_predictor(), threshold=1.0, discrepancy=constant)
    result = custom.score(latents)

    assert 16 < default.evaluated_regions < 16 + 196
    assert result.evaluated_regions == default.evaluated_regions
    assert result.coarse_scores.max() == huge
    assert result.raw_image_score >= huge


def test_image_score_is_undefined_for_custom_discrepancy() -> None:
    scorer = MaskedLatentScorer(_predictor(), discrepancy=layer_norm_cosine_discrepancy)
    result = scorer.score(_latents(), mode="fixed")

    assert math.isinf(result.score_upper_bound)
    assert result.raw_image_score >= 0
    with pytest.raises(ValueError, match="raw_image_score"):
        _ = result.image_score


def test_window_predictions_cover_only_requested_windows() -> None:
    scorer = MaskedLatentScorer(_predictor())
    latents = _latents()
    windows = scorer.coarse.windows()[:2]

    predictions = scorer.window_predictions(latents, windows)
    covered = sorted(i for w in windows for i in w.token_indices)
    other = [i for i in range(196) if i not in covered]

    assert predictions.shape == (196, DIM)
    assert predictions[covered].abs().sum() > 0
    assert predictions[other].abs().sum() == 0


def test_layer_norm_cosine_is_pearson_and_affine_invariant() -> None:
    generator = torch.Generator().manual_seed(0)
    actual = torch.randn(2, 5, 32, generator=generator)
    predicted = torch.randn(2, 5, 32, generator=generator)

    base = layer_norm_cosine_discrepancy(predicted, actual)
    shifted = layer_norm_cosine_discrepancy(3.0 * predicted + 7.0, 0.5 * actual - 2.0)
    centred = predicted - predicted.mean(-1, keepdim=True)
    other = actual - actual.mean(-1, keepdim=True)
    pearson = 1 - F.cosine_similarity(centred, other, dim=-1)

    assert torch.allclose(base, shifted, atol=1e-4)
    assert torch.allclose(base, pearson, atol=1e-4)
    assert layer_norm_cosine_discrepancy(actual, actual).abs().max() < 1e-5


def test_channel_stats_match_pooled_mean_and_std() -> None:
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(7, 10, 4, generator=generator) * torch.tensor(
        [1.0, 2.0, 3.0, 4.0]
    ) + torch.tensor([5.0, 0.0, -1.0, 2.0])

    stats = fit_channel_stats(latents)
    flat = latents.reshape(-1, 4)

    assert torch.allclose(stats.mean, flat.mean(0), atol=1e-5)
    assert torch.allclose(stats.std, flat.std(0), atol=1e-5)
    with pytest.raises(ValueError, match="at least two"):
        fit_channel_stats(torch.zeros(1, 4))


def test_channel_standardisation_downweights_an_outlier_channel() -> None:
    # Channel 0 carries a huge constant offset that dominates raw cosine.
    stats = ChannelStats(mean=torch.tensor([100.0, 0.0]), std=torch.tensor([1.0, 1.0]))
    actual = torch.tensor([[[100.0, 1.0]]])
    predicted = torch.tensor([[[100.0, -1.0]]])

    raw = cosine_discrepancy(predicted, actual)
    standardised = ChannelStandardisedCosine(stats)(predicted, actual)

    assert raw.item() < 1e-3  # the offset hides the disagreement
    assert standardised.item() == pytest.approx(2.0, abs=1e-5)  # now fully opposed


def test_channel_standardisation_with_identity_stats_equals_cosine() -> None:
    generator = torch.Generator().manual_seed(0)
    actual = torch.randn(3, 4, 8, generator=generator)
    predicted = torch.randn(3, 4, 8, generator=generator)
    identity = ChannelStats(mean=torch.zeros(8), std=torch.ones(8))

    assert torch.allclose(
        ChannelStandardisedCosine(identity)(predicted, actual),
        cosine_discrepancy(predicted, actual),
    )


def test_mahalanobis_scores_nominal_near_one_and_outliers_high() -> None:
    generator = torch.Generator().manual_seed(0)
    scales = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.1])
    train = torch.randn(4000, 8, generator=generator) * scales
    held_out = torch.randn(2000, 8, generator=generator) * scales

    model = fit_mahalanobis_residual(train, n_components=4, ridge=0.0)
    zero = torch.zeros(held_out.shape)
    nominal = model(zero, held_out).mean()
    # Same Euclidean norm as a nominal residual, but along a variance-free axis.
    outlier_axis = torch.zeros(1, 8)
    outlier_axis[0, 7] = 3.0
    on_signal_axis = torch.zeros(1, 8)
    on_signal_axis[0, 0] = 3.0

    assert nominal.item() == pytest.approx(1.0, abs=0.1)
    assert model.components.T @ model.components == pytest.approx(
        torch.eye(4), abs=1e-4
    )
    assert model(torch.zeros(1, 8), 5 * torch.randn(1, 8) * scales).item() > 5
    # Directions outside the retained subspace are ignored by construction.
    assert model(torch.zeros(1, 8), outlier_axis).item() < 0.2
    assert model(torch.zeros(1, 8), on_signal_axis).item() < 1.0  # 3 sigma / 4 comps


def test_mahalanobis_uses_predictor_residual_not_cosine() -> None:
    generator = torch.Generator().manual_seed(0)
    train = torch.randn(500, 6, generator=generator)
    model = fit_mahalanobis_residual(train, n_components=3)
    actual = torch.randn(1, 2, 6, generator=generator)

    perfect = model(actual, actual)
    off = model(actual + 10.0, actual)

    assert perfect.shape == (1, 2)
    assert off.min() > perfect.max()


def test_mahalanobis_rejects_degenerate_fits() -> None:
    with pytest.raises(ValueError, match="n_components"):
        fit_mahalanobis_residual(torch.randn(10, 20), n_components=20)
    with pytest.raises(ValueError, match="n_components"):
        fit_mahalanobis_residual(torch.randn(10, 20), n_components=10)
    with pytest.raises(ValueError, match="degenerate"):
        fit_mahalanobis_residual(torch.zeros(50, 6), n_components=2)
    with pytest.raises(ValueError, match="ridge"):
        fit_mahalanobis_residual(torch.randn(50, 6), n_components=2, ridge=-1.0)


def test_position_normaliser_zero_mean_unit_std_on_fit_data() -> None:
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(30, 196, generator=generator) * torch.linspace(0.5, 3, 196)

    normaliser = PositionNormaliser.fit(scores, std_floor_fraction=0.0)
    z = normaliser.apply(scores)

    assert torch.allclose(z.mean(0), torch.zeros(196), atol=1e-4)
    assert torch.allclose(z.std(0), torch.ones(196), atol=1e-3)


def test_position_normaliser_floors_tiny_stds_and_supports_centring_only() -> None:
    scores = torch.rand(20, 196) + 1.0
    scores[:, 7] = 2.5  # a position with zero spread
    floored = PositionNormaliser.fit(scores, std_floor_fraction=0.5)
    centred = PositionNormaliser.fit(scores, divide=False)

    assert floored.std[7] >= 0.5 * floored.std.median() - 1e-6
    assert floored.std[7] > 0
    assert torch.isfinite(floored.apply(scores)).all()
    assert torch.equal(centred.std, torch.ones(196))
    assert torch.allclose(centred.apply(scores), scores - scores.mean(0), atol=1e-5)
    with pytest.raises(ValueError, match="N >= 2"):
        PositionNormaliser.fit(scores[:1])


def test_smoothing_sigma_zero_is_identity_and_constants_are_preserved() -> None:
    maps = torch.rand(2, 1, 32, 32)
    constant = torch.full((1, 1, 32, 32), 3.0)

    assert smooth_score_maps(maps, 0.0) is maps
    assert torch.allclose(smooth_score_maps(constant, 2.0), constant, atol=1e-5)


def test_smoothing_impulse_is_a_normalised_symmetric_gaussian() -> None:
    impulse = torch.zeros(1, 1, 41, 41)
    impulse[0, 0, 20, 20] = 1.0

    blurred = smooth_score_maps(impulse, 2.0)[0, 0]
    expected = torch.exp(-0.5 * (torch.arange(-20, 21) / 2.0) ** 2)
    expected = expected / expected.sum()

    assert blurred.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert torch.allclose(blurred[20], expected * expected[20], atol=1e-5)
    assert torch.allclose(blurred, blurred.T, atol=1e-6)
    assert torch.allclose(blurred, blurred.flip(0), atol=1e-6)


def test_smoothing_reduces_a_single_spike_and_validates_inputs() -> None:
    spike = torch.zeros(1, 1, 64, 64)
    spike[0, 0, 32, 32] = 10.0

    assert smooth_score_maps(spike, 4.0).max() < 10.0
    with pytest.raises(ValueError, match="too large"):
        smooth_score_maps(spike, 20.0)
    with pytest.raises(ValueError, match="non-negative"):
        smooth_score_maps(spike, -1.0)
    with pytest.raises(ValueError, match="shape"):
        smooth_score_maps(torch.zeros(64, 64), 1.0)
