# Colored MNIST – Shortcut Learning Interactive Demo

An isolated, inference-only Streamlit application for comparing three trained `SmallCNN` checkpoints (`p = 0.99`, `seed = 0`) on the **same underlying MNIST test digit (index 514, true label `6`)** under three color conditions:

- **Grayscale** (`digit_6_grayscale.png`, no color cue)
- **Aligned Color** (`digit_6_aligned.png`, Orange — Class `6` training color)
- **Conflicting Color** (`digit_6_conflicting.png`, Purple — Class `7` training color)

## Checkpoints Loaded (Read-Only)

All weights are loaded directly from the existing repository checkpoints (relative to the repository root):
- **Baseline**: `checkpoints/none_p099_seed0.pth`
- **Data Mitigation**: `checkpoints/augmentation_p099_seed0.pth`
- **Reweighting**: `checkpoints/reweighting_p099_seed0.pth`

No training, fine-tuning, or dataset downloading is performed by the app.

## Local Launch

From the repository root (`aai2-shortcut-learning/`):

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

## Deploying on Streamlit Community Cloud

1. **Ensure the 3 checkpoints are present in your GitHub repository**:
   By default, `.gitignore` ignores `*.pth` files. Before deploying, make sure these three ~429 KB files are committed and pushed to your GitHub repo (e.g., via `git add -f checkpoints/none_p099_seed0.pth checkpoints/augmentation_p099_seed0.pth checkpoints/reweighting_p099_seed0.pth` or `!checkpoints/...` exceptions in `.gitignore`).
2. **Push `demo/`** (`app.py`, `model_loader.py`, `inference.py`, `requirements.txt`, `README.md`, `examples/*.png`) and `src/models.py` to your main GitHub branch.
   *(Do not commit `demo/.venv/` if you created a local virtual environment.)*
3. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and select:
   - **Repository**: your GitHub repository
   - **Branch**: `main` (or your target branch)
   - **Main file path**: `demo/app.py`
4. Streamlit Community Cloud will automatically detect `demo/requirements.txt` and launch the app.

## Troubleshooting

- **`FileNotFoundError` for checkpoints on Streamlit Cloud**: Verify that the three `.pth` files in `checkpoints/` are actually committed in GitHub (not excluded by `.gitignore`).
- **`ModuleNotFoundError: No module named 'src'`**: `demo/model_loader.py` automatically adds the repository root to `sys.path`; ensure `demo/` and `src/` are both at the repository root.
