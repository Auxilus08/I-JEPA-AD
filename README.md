# I-JEPA-AD

Research codebase for decoder-free latent anomaly detection with I-JEPA.

The real-model pilot is implemented and evaluated on the available MVTec AD Bottle,
Capsule, and Hazelnut categories: frozen ViT-g/16 encoding, resumable latent caches,
cosine/L2 predictor training, adaptive/fixed/dense I-JEPA inference, anomaly metrics,
and validation-calibrated decision thresholds. The Streamlit frontend defaults to real
inference; synthetic mock mode is explicit. Pilot results are not full-benchmark claims:
the other 12 categories and valid PatchCore/MAE comparisons remain outstanding. See
[TRANSFER.md](TRANSFER.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for measured
results and limitations.

## Development

```bash
python -m pip install -e ".[dev]"
pre-commit install
ruff check .
black --check .
mypy src tests
pytest tests/unit tests/integration
```

For the dashboard, install the optional frontend dependencies and run from the project
root:

```bash
python -m pip install -e ".[frontend]"
python -m streamlit run frontend/app.py
```

Real image inference requires the local I-JEPA checkpoint and a trained predictor.
Synthetic mock mode must be selected explicitly in the UI.

The test suite requires a PyTorch installation that can import successfully on the
host. The CI workflow uses Python 3.11 and installs the project dependencies on an
Ubuntu runner; CUDA validation is not part of CI and is documented in
[`docs/GPU_VALIDATION_CHECKLIST.md`](docs/GPU_VALIDATION_CHECKLIST.md).
