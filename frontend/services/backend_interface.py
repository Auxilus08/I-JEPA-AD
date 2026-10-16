"""Person 4 Backend Application Interface.

Conforms to the Anomaly-detection-to-frontend contract specified on Pages 5 & 8
of the project design document.

Provides:
- Stable public `run_inference(...)` entrypoint.
- Explicit mock/real execution indicator: `is_mock: bool`.
- Consistent region detection aligned with the evaluated anomaly score.
- Dynamic image-dependent heatmap synthesis when checkpoints are unmounted.
"""

from __future__ import annotations

import importlib.util
import json
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ijepa_ad.inference import InferenceMode

PYTORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("ijepa_ad") is not None
)
REAL_METHODS: dict[str, InferenceMode] = {
    "adaptive_ijepa": "adaptive",
    "dense_masked_ijepa": "dense",
    "fixed_mask_ijepa": "fixed",
}


def _validate_image_input(image: Any) -> Image.Image:
    """Validate and ensure image is a valid PIL Image instance."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise ValueError(
        f"Expected image to be a PIL.Image instance, received {type(image).__name__}."
    )


def _compute_image_signature(image: Image.Image) -> tuple[float, float, float]:
    """Derive deterministic spatial parameters from actual image pixel statistics."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=-1)
    h, w = gray.shape

    # Find region with largest local variance / gradient
    grad_y, grad_x = np.gradient(gray)
    mag = np.hypot(grad_y, grad_x)

    # Coordinates of maximum gradient/defect center
    max_idx = int(np.argmax(mag))
    cy, cx = np.unravel_index(max_idx, (h, w))
    var_intensity = float(np.clip(mag.mean() * 4.0 + gray.std() * 1.5, 0.15, 0.95))

    return float(cx), float(cy), var_intensity


def _generate_mock_inference(
    image: Image.Image,
    method: str,
    category: str | None,
    anomaly_threshold: float = 0.50,
) -> dict[str, Any]:
    """Generate image-dependent synthetic results for testing and demo."""
    start_time = time.perf_counter()
    w, h = image.size

    cx, cy, intensity = _compute_image_signature(image)
    radius = max(min(h, w) // 6, 12)

    # Image-dependent 2D Gaussian heatmap centered around the highest-contrast region
    grid_y, grid_x = np.ogrid[:h, :w]
    dist_sq = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
    raw_heat = np.exp(-dist_sq / (2.0 * (radius**2))) * intensity

    # Method-specific calibration
    method_multipliers = {
        "adaptive_ijepa": 1.0,
        "dense_masked_ijepa": 1.02,
        "fixed_mask_ijepa": 0.92,
        "patchcore": 0.98,
        "mae_reconstruction": 0.94,
    }
    multiplier = method_multipliers.get(method, 1.0)
    calibrated_heat = np.clip(raw_heat * multiplier, 0.0, 1.0).astype(np.float32)

    image_score = float(np.percentile(calibrated_heat, 99.0))
    is_anomalous = image_score > anomaly_threshold

    # Mask counts per method specification
    mask_counts = {
        "adaptive_ijepa": 14,
        "dense_masked_ijepa": 196,
        "fixed_mask_ijepa": 16,
        "patchcore": 0,
        "mae_reconstruction": 1,
    }
    total_regions = {
        "adaptive_ijepa": 212,
        "dense_masked_ijepa": 196,
        "fixed_mask_ijepa": 16,
        "patchcore": 0,
        "mae_reconstruction": 1,
    }

    # Consistency rule: Nominal samples NEVER produce defect bounding boxes
    detected_regions: list[dict[str, Any]] = []
    if is_anomalous:
        box = [
            max(0, int(cx - radius)),
            max(0, int(cy - radius)),
            min(w, int(cx + radius)),
            min(h, int(cy + radius)),
        ]
        detected_regions.append(
            {
                "box": box,
                "confidence": round(image_score, 3),
                "label": f"Defect ({category or 'generic'})",
            }
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "image_score": round(image_score, 4),
        "pixel_heatmap": calibrated_heat,
        "overlay": None,
        "detected_regions": detected_regions,
        "mask_count": mask_counts.get(method, 14),
        "total_regions": total_regions.get(method, 212),
        "latency_ms": round(latency_ms, 2),
        "peak_vram_mb": 312.4,
        "method": method,
        "category": category,
        "is_mock": True,
        "status": "mock",
    }


DEFAULT_PREDICTOR_DIR = "checkpoints/predictors"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_predictor_dir(root: Path, predictor_dir: str | Path | None = None) -> Path:
    """Directory holding ``<category>_cosine.pt``; relative paths are under ``root``."""
    path = Path(DEFAULT_PREDICTOR_DIR if predictor_dir is None else predictor_dir)
    return path if path.is_absolute() else root / path


@lru_cache(maxsize=1)
def _checkpoint_sha256(root: str) -> str:
    checkpoint = Path(root) / "checkpoints" / "IN22K-vit.g.16-600e.pth.tar"
    stat = checkpoint.stat()
    cache_path = checkpoint.with_name(checkpoint.name + ".sha256.json")
    cached: Any = None
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as exc:
        warnings.warn(
            f"Ignoring invalid encoder hash cache {cache_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    if (
        isinstance(cached, dict)
        and cached.get("size") == stat.st_size
        and cached.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(cached.get("sha256"), str)
        and len(cached["sha256"]) == 64
        and all(char in "0123456789abcdef" for char in cached["sha256"])
    ):
        return cached["sha256"]

    from ijepa_ad.caching.latent_cache import compute_file_sha256

    digest = str(compute_file_sha256(checkpoint))
    try:
        cache_path.write_text(
            json.dumps(
                {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": digest,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        warnings.warn(
            f"Could not persist encoder hash cache {cache_path}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    return digest


@lru_cache(maxsize=1)
def _load_encoder(root: str, device_name: str) -> Any:
    from ijepa_ad.encoder.ijepa_wrapper import IJEPAEncoder

    project = Path(root)
    checkpoint = project / "checkpoints" / "IN22K-vit.g.16-600e.pth.tar"
    official_repo = project / "external" / "ijepa"
    return IJEPAEncoder(
        checkpoint,
        official_repo_path=official_repo,
        device=device_name,
    )


@lru_cache(maxsize=6)
def _load_predictor(
    root: str, category: str, device_name: str, predictor_dir: str
) -> Any:
    from ijepa_ad.caching.cache_schema import is_safe_name
    from ijepa_ad.inference import load_predictor_checkpoint

    if not is_safe_name(category):
        raise ValueError(f"invalid MVTec category: {category!r}")
    path = Path(predictor_dir) / f"{category}_cosine.pt"
    if not path.is_file():
        raise FileNotFoundError(f"trained cosine predictor not found: {path}")
    return load_predictor_checkpoint(
        path,
        category=category,
        encoder_sha256=_checkpoint_sha256(root),
        device=device_name,
    )


def detected_regions_for_threshold(
    heatmap: np.ndarray[Any, Any],
    image_size: tuple[int, int],
    image_score: float,
    anomaly_threshold: float,
    category: str,
) -> list[dict[str, Any]]:
    if image_score <= anomaly_threshold or float(heatmap.max()) == float(heatmap.min()):
        return []
    minimum = float(heatmap.min())
    cutoff = float(np.quantile(heatmap, 0.95))
    selected = heatmap > minimum if cutoff <= minimum else heatmap >= cutoff
    width, height = image_size
    scale_x, scale_y = width / heatmap.shape[1], height / heatmap.shape[0]
    regions = []
    for min_row, min_column, max_row, max_column in _component_bounds(selected):
        regions.append(
            {
                "box": [
                    max(0, int(min_column * scale_x)),
                    max(0, int(min_row * scale_y)),
                    min(width, int(max_column * scale_x)),
                    min(height, int(max_row * scale_y)),
                ],
                "confidence": round(image_score, 3),
                "label": f"Anomaly ({category})",
            }
        )
    return regions


def _component_bounds(mask: np.ndarray[Any, Any]) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros(mask.size, dtype=bool)
    bounds: list[tuple[int, int, int, int]] = []

    for start in np.flatnonzero(mask):
        if visited[start]:
            continue
        visited[start] = True
        stack = [int(start)]
        min_row = max_row = int(start) // width
        min_column = max_column = int(start) % width
        while stack:
            index = stack.pop()
            row, column = divmod(index, width)
            min_row = min(min_row, row)
            max_row = max(max_row, row)
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            neighbors = []
            if row:
                neighbors.append(index - width)
            if row + 1 < height:
                neighbors.append(index + width)
            if column:
                neighbors.append(index - 1)
            if column + 1 < width:
                neighbors.append(index + 1)
            for neighbor in neighbors:
                if mask.flat[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        bounds.append((min_row, min_column, max_row + 1, max_column + 1))
    return bounds


def _run_real_inference(
    image: Image.Image,
    method: str,
    category: str | None,
    anomaly_threshold: float,
    predictor_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not PYTORCH_AVAILABLE:
        raise RuntimeError(
            "Real inference requires both PyTorch and the ijepa_ad package; "
            "install the project dependencies or enable synthetic mock mode."
        )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Real inference requires PyTorch; install the project dependencies "
            "or enable synthetic mock mode."
        ) from exc

    from ijepa_ad.caching.cache_schema import is_safe_name
    from ijepa_ad.data.transforms import preprocess_image
    from ijepa_ad.inference import MaskedLatentScorer, load_masking_settings

    if category is None:
        raise ValueError("category is required for real I-JEPA inference")
    if method not in REAL_METHODS:
        raise NotImplementedError(f"real inference is not implemented for {method!r}")
    if not is_safe_name(category):
        raise ValueError(f"invalid MVTec category: {category!r}")

    root = _project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = _load_encoder(str(root), str(device))
    predictor = _load_predictor(
        str(root),
        category,
        str(device),
        str(resolve_predictor_dir(root, predictor_dir)),
    )
    image_tensor = preprocess_image(image).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    latents = encoder.encode(image_tensor)
    masking_settings = load_masking_settings(
        root / "configs" / "masking" / "coarse_to_fine.yaml"
    )
    scores = MaskedLatentScorer(predictor, **masking_settings).score(
        latents, mode=REAL_METHODS[method]
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_vram_mb = None

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    heatmap = scores.heatmap[0, 0].detach().cpu().numpy().astype(np.float32)
    image_score = scores.image_score
    return {
        "image_score": image_score,
        "pixel_heatmap": heatmap,
        "overlay": None,
        "detected_regions": detected_regions_for_threshold(
            heatmap, image.size, image_score, anomaly_threshold, category
        ),
        "mask_count": scores.evaluated_regions,
        "total_regions": scores.total_regions,
        "latency_ms": latency_ms,
        "peak_vram_mb": peak_vram_mb,
        "method": method,
        "category": category,
        "is_mock": False,
        "status": "real",
    }


def run_inference(
    image: Any,
    method: str = "adaptive_ijepa",
    category: str | None = None,
    use_mock: bool = False,
    anomaly_threshold: float = 0.50,
    predictor_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run actual inference unless a synthetic response is explicitly requested.

    ``predictor_dir`` selects the checkpoint directory (default
    ``checkpoints/predictors``; notebook 02 writes
    ``checkpoints/predictors_augmented``).
    """
    validated_img = _validate_image_input(image)
    if not 0.0 <= anomaly_threshold <= 1.0:
        raise ValueError("anomaly_threshold must be in [0, 1]")
    if use_mock:
        return _generate_mock_inference(
            validated_img,
            method=method,
            category=category,
            anomaly_threshold=anomaly_threshold,
        )
    return _run_real_inference(
        validated_img, method, category, anomaly_threshold, predictor_dir
    )
