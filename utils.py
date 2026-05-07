import ultralytics
import pandas as pd
import streamlit as st

def lr_coefficients_styler(top_features):
    """
    Compose a styled table to display L1 logistic regression coefficients.

    Parameters
    ----------
    top_features: List[Tuple[str, float]]
        List of (feature_name, coefficient)'s.
    
    Returns
    -------
    Pandas Styler object
    """
    
    df = pd.DataFrame({
        "#": range(1, len(top_features) + 1),
        "Feature": [t[0] for t in top_features],
        "β (L1 coefficient)": [float(t[1]) for t in top_features],
        "Lean": ["Fastball" if float(t[1]) > 0 else "Offspeed" for t in top_features],
    })

    coef = df["β (L1 coefficient)"]
    lim = float(coef.abs().max()) if len(coef) else 1.0
    if lim < 1e-12:
        lim = 1.0

    return (
        df.style.format({"β (L1 coefficient)": "{:+,.4f}"})
        .background_gradient(subset=["β (L1 coefficient)"], cmap="RdBu_r", vmin=-lim, vmax=lim)
        .set_properties(**{"text-align": "left"}, subset=["Feature", "Lean"])
        .set_properties(**{"font-family": "'IBM Plex Mono', monospace"},
                        subset=["#", "β (L1 coefficient)"])
    )

@st.cache_resource
def load_models():
    """Load YOLO pose and glove bounding box estimation models."""
    return ultralytics.YOLO("yolov8x-pose.pt"), ultralytics.YOLO("GLOVE_MODEL.pt")

