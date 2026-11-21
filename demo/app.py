"""Interactive Streamlit demonstration for Colored MNIST Shortcut Learning."""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from inference import load_example_image, predict
from model_loader import CHECKPOINT_PATHS, MODEL_DESCRIPTIONS, load_all_models

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(DEMO_DIR, "examples")
TRUE_LABEL = 6

EXAMPLES = {
    "Grayscale": {
        "path": os.path.join(EXAMPLES_DIR, "digit_6_grayscale.png"),
        "color_info": "None (white / grayscale)",
        "description": "Same MNIST test digit (index 514) with no color cue (`test_clean`).",
    },
    "Aligned Color": {
        "path": os.path.join(EXAMPLES_DIR, "digit_6_aligned.png"),
        "color_info": "Orange (associated with Class 6 in training)",
        "description": "Same digit painted with its training-aligned shortcut color (`test_aligned`).",
    },
    "Conflicting Color": {
        "path": os.path.join(EXAMPLES_DIR, "digit_6_conflicting.png"),
        "color_info": "Purple (associated with Class 7 in training)",
        "description": "Same digit painted with a conflicting shortcut color (`test_conflicting`).",
    },
}


@st.cache_resource
def get_models():
    """Load all three checkpoints once and cache them across reruns."""
    return load_all_models()


@st.cache_data
def get_example(condition: str):
    """Load and cache the preprocessed example image and tensor."""
    return load_example_image(EXAMPLES[condition]["path"])


def plot_probabilities(probs: list[float], pred_label: int, is_correct: bool):
    """Create a clean 10-class softmax probability bar chart."""
    fig, ax = plt.subplots(figsize=(4.4, 2.35))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    digits = np.arange(10)
    bar_colors = ["#cbd5e1"] * 10
    bar_colors[pred_label] = "#16a34a" if is_correct else "#dc2626"

    ax.bar(digits, probs, color=bar_colors, width=0.72, edgecolor="none")
    ax.set_xticks(digits)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Probability", fontsize=8.5, color="#475569")
    ax.set_xlabel("Digit Class", fontsize=8.5, color="#475569")
    ax.tick_params(labelsize=8, colors="#475569")
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    fig.tight_layout()
    return fig


def render_model_card(model_name: str, result: dict):
    """Render a structured prediction card for one model."""
    is_ok = result["is_correct"]
    badge_bg = "#dcfce7" if is_ok else "#fee2e2"
    badge_fg = "#166534" if is_ok else "#991b1b"
    status_text = "✓ Correct" if is_ok else "✗ Incorrect"

    with st.container(border=True):
        st.subheader(model_name)
        st.caption(MODEL_DESCRIPTIONS[model_name])

        c1, c2 = st.columns(2)
        c1.metric("Predicted Digit", str(result["predicted_label"]))
        c2.metric("Confidence", f"{result['confidence']:.1%}")

        st.markdown(
            f"<div style='padding:6px 10px;border-radius:6px;background:{badge_bg};"
            f"color:{badge_fg};font-weight:600;text-align:center;margin-bottom:8px;'>"
            f"{status_text} (True Label: {result['true_label']})</div>",
            unsafe_allow_html=True,
        )

        fig = plot_probabilities(result["probabilities"], result["predicted_label"], is_ok)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def main():
    st.set_page_config(
        page_title="Colored MNIST – Shortcut Learning Demonstration",
        layout="wide",
    )

    st.title("Colored MNIST – Shortcut Learning Demonstration")
    st.markdown("#### *Comparing Baseline, Data-Level Mitigation and Weight-Level Reweighting.*")
    st.write(
        "*The same handwritten digit can produce different predictions depending on the "
        "visual shortcuts learned by the model.*"
    )
    st.divider()

    # Load cached models (inference only, no training or downloads)
    models = get_models()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        condition = st.radio(
            "Input Image Condition",
            options=list(EXAMPLES.keys()),
            index=2,
        )
        mode = st.radio(
            "Model Mode",
            options=["Compare All Models", "Single Model"],
            index=0,
        )
        selected_model = None
        if mode == "Single Model":
            selected_model = st.selectbox("Select Model", options=list(CHECKPOINT_PATHS.keys()))

        st.divider()
        st.caption(
            "Illustrative single-sample demo (MNIST test index 514, p = 0.99, seed 0). "
            "Overall benchmark conclusions are reported in `results/summary.md`."
        )

    # Top row: selected input image + metadata
    pil_img, img_tensor = get_example(condition)
    meta = EXAMPLES[condition]

    col_img, col_info = st.columns([1, 2.2])
    with col_img:
        display_img = pil_img.resize((224, 224), resample=Image.NEAREST)
        st.image(display_img, caption=f"{condition} (28×28 input, upscaled for display)")
    with col_info:
        st.markdown(f"### Selected Input: **{condition}**")
        st.markdown(f"- **True Label:** `{TRUE_LABEL}`")
        st.markdown(f"- **Color Condition:** {meta['color_info']}")
        st.markdown(f"- **Description:** {meta['description']}")

    st.divider()

    # Prediction results
    if mode == "Compare All Models":
        cols = st.columns(3)
        for col, model_name in zip(cols, CHECKPOINT_PATHS.keys()):
            with col:
                res = predict(models[model_name], img_tensor, true_label=TRUE_LABEL)
                render_model_card(model_name, res)
    else:
        _, center_col, _ = st.columns([1, 1.4, 1])
        with center_col:
            res = predict(models[selected_model], img_tensor, true_label=TRUE_LABEL)
            render_model_card(selected_model, res)


if __name__ == "__main__":
    main()
