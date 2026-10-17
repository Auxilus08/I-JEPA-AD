"""Evaluation output records predictor provenance and the image-score definition."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from ijepa_ad.evaluation import runner
from ijepa_ad.inference import InferenceMode, MaskingSettings
from ijepa_ad.predictor.transformer_predictor import (
    LatentTransformerPredictor,
    PredictorConfig,
)

ENCODER_SHA = "a" * 64
SETTINGS: MaskingSettings = {
    "coarse_size": 4,
    "fine_size": 1,
    "threshold": 0.5,
    "lam": 0.5,
}


def _write_checkpoint(path: Path, **extra: object) -> None:
    config = PredictorConfig(
        embedding_dim=1408,
        num_tokens=196,
        hidden_dim=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    payload: dict[str, object] = {
        "model_state_dict": LatentTransformerPredictor(config).state_dict(),
        "model_config": config.to_dict(),
        "training_config": {"loss_type": "cosine"},
        "cache_provenance": {"category": "bottle", "checkpoint_sha256": ENCODER_SHA},
    }
    payload.update(extra)
    torch.save(payload, path)


def _annotation(tmp_path: Path, name: str, defect: bool) -> SimpleNamespace:
    image_path = tmp_path / f"{name}.png"
    Image.new("RGB", (32, 32)).save(image_path)
    mask_path = None
    if defect:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:16, 8:16] = 255
        mask_path = tmp_path / f"{name}_mask.png"
        Image.fromarray(mask).save(mask_path)
    return SimpleNamespace(
        image_path=image_path, mask_path=mask_path, label="crack" if defect else "good"
    )


def _patch_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(0)

    def sample(image_id: str, annotation: Any = None) -> SimpleNamespace:
        latent = torch.randn(196, 1408, generator=generator)
        return SimpleNamespace(image_id=image_id, latent=latent, annotation=annotation)

    val = [sample(f"val-{index}") for index in range(3)]
    test = [
        sample("test-good", _annotation(tmp_path, "good", defect=False)),
        sample("test-bad", _annotation(tmp_path, "bad", defect=True)),
    ]
    monkeypatch.setattr(
        runner,
        "validate_category_splits",
        lambda *_: (SimpleNamespace(samples=()), SimpleNamespace(samples=test)),
    )
    monkeypatch.setattr(
        runner,
        "split_nominal_train_val",
        lambda *_, **__: SimpleNamespace(val=val),
    )
    monkeypatch.setattr(
        runner,
        "load_cached_latent_dataset",
        lambda _root, expected, **__: val if expected.split == "val" else test,
    )


def _evaluate(tmp_path: Path, mode: InferenceMode = "adaptive") -> dict[str, Any]:
    return runner.evaluate_category(
        category="bottle",
        dataset_root=tmp_path,
        cache_root=tmp_path,
        predictor_path=tmp_path / "bottle_cosine.pt",
        encoder_sha256=ENCODER_SHA,
        device="cpu",
        mode=mode,
        masking_settings=SETTINGS,
    )


def test_result_records_current_checkpoint_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_data(monkeypatch, tmp_path)
    training_config = {"loss_type": "cosine", "mask_strategy": "inference_aligned"}
    reproducibility = {"seed": 3, "git_commit": "abc123", "git_dirty": False}
    _write_checkpoint(
        tmp_path / "bottle_cosine.pt",
        training_config=training_config,
        reproducibility=reproducibility,
    )

    result = _evaluate(tmp_path)

    provenance = result["predictor_provenance"]
    assert provenance["status"] == "complete"
    assert provenance["mask_strategy"] == "inference_aligned"
    assert provenance["training_config"] == training_config
    assert provenance["reproducibility"] == reproducibility
    assert result["predictor_checkpoint"] == str(tmp_path / "bottle_cosine.pt")
    json.dumps(result)  # the whole record must stay JSON-serializable


def test_result_marks_legacy_checkpoint_without_inventing_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_data(monkeypatch, tmp_path)
    _write_checkpoint(tmp_path / "bottle_cosine.pt")

    result = _evaluate(tmp_path)

    provenance = result["predictor_provenance"]
    assert provenance["status"] == "legacy"
    assert provenance["mask_strategy"] is None
    assert provenance["reproducibility"] is None
    json.dumps(result)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("mode", "bound"), [("fixed", 2.0), ("dense", 1.0), ("adaptive", 3.0)]
)
def test_result_records_image_score_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: InferenceMode,
    bound: float,
) -> None:
    _patch_data(monkeypatch, tmp_path)
    _write_checkpoint(tmp_path / "bottle_cosine.pt")

    result = _evaluate(tmp_path, mode)

    definition = result["image_score_definition"]
    assert definition["upper_bound"] == bound  # lambda = 0.5
    assert definition["lambda"] == 0.5
    assert definition["mode"] == mode
    assert 0.0 <= result["decision_threshold"] <= 1.0


def test_encoder_mismatch_still_blocks_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_data(monkeypatch, tmp_path)
    _write_checkpoint(tmp_path / "bottle_cosine.pt")

    with pytest.raises(ValueError, match="different encoder"):
        runner.evaluate_category(
            category="bottle",
            dataset_root=tmp_path,
            cache_root=tmp_path,
            predictor_path=tmp_path / "bottle_cosine.pt",
            encoder_sha256="b" * 64,
            device="cpu",
            mode="fixed",
            masking_settings=SETTINGS,
        )
