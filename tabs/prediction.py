import os
import numpy as np
import pandas as pd
import streamlit as st
import processing.tab1_helper as t1
import processing.tab3_helper as t3

from utils import load_models
from ui.components import info_panel, warn_panel, success_panel, section_label

def render_prediction_tab(base_dir):
    """
    Render Tab 4 (Pitch Prediction).

    Upload a new pitch video, run motion estimation,
    extract features, and predict pitch class (fastball / offspeed)
    using Tab 3's models, and displays top contributing features.

    Parameters
    ----------
    base_dir: str
        Root directory for Pitches/ and Keypoints/ folders.
    """

    ### TOP INSTRUCTION PANEL
    st.markdown(
        info_panel(
            "Upload a new <code>.MOV</code> or <code>.mp4</code> video of an"
            " unlabelled pitch. With our trained models from Tab 03, we can"
            " compute its motion (this may require a few minutes) and predict its class."
        ),
        unsafe_allow_html=True,
    )

    ### ENSURE MODELS FIT (TAB 3 BEFORE TAB 4)
    if ("single_pitcher_clf" not in st.session_state)\
         or ("single_pitcher_scaler" not in st.session_state):
        st.markdown(
            warn_panel("No trained model found: visit Tab 03 to train."),
            unsafe_allow_html=True,
        )
        return

    ### BUTTON TO UPLOAD NEW VIDEO
    pred_video = st.file_uploader(
        "Upload A New Pitch Video", type=["mov", "mp4"], key="pred_upload"
    )
    if pred_video is not None:
        col_pred1, col_pred2 = st.columns([1, 1], gap="large")
        with col_pred1:
            st.video(pred_video)

        ### BUTTON TO RUN PITCH PREDICTION
        with col_pred2:
            if st.button("▶ Run Pitch Prediction", type="primary", width="stretch"):
                pred_vid_target = os.path.join(base_dir, "Pitches", "prediction.mov")
                pred_kpt_target = os.path.join(base_dir, "Keypoints", "prediction.npy")
                os.makedirs(os.path.dirname(pred_vid_target), exist_ok=True)

                with open(pred_vid_target, "wb") as f:
                    f.write(pred_video.getbuffer())

                progress_bar = st.progress(0)
                status_text = st.empty()

                ### PERFORM MOTION ESTIMATION
                with st.spinner("Initializing models:"):
                    pose_m, glove_m = load_models()
                    success, meta = t1.processVideo(
                        pred_vid_target, pred_kpt_target, pose_m, glove_m, progress_bar, status_text
                    )

                if success:
                    progress_bar.empty()
                    status_text.empty()
                    seq = np.load(pred_kpt_target, allow_pickle=True)
                    if isinstance(seq, np.ndarray) and seq.dtype == object:
                        seq = seq.tolist()

                    try:
                        ### COMPUTE KEYPOINT FEATURE REPRESENTATIONS
                        raw_feats = t3.extractFeatures(seq, label="Unknown")
                        if len(raw_feats) <= 1:
                            st.error("Extracted sequence was empty or malformed.")

                        ### PERFORM MODELING (PREDICT NEW PITCH CLASS)
                        else:
                            clf = st.session_state["single_pitcher_clf"]
                            scaler = st.session_state["single_pitcher_scaler"]
                            feature_names = st.session_state["single_pitcher_features"]

                            raw_df = pd.DataFrame([raw_feats])
                            X_test_raw = raw_df[feature_names].fillna(0).values
                            X_test_scaled = scaler.transform(X_test_raw)

                            pred = clf.predict(X_test_scaled)[0]
                            probs = clf.predict_proba(X_test_scaled)[0]

                            pred_class = "Fastball" if pred == 1 else "Offspeed"
                            max_prob = max(probs) * 100

                            ### DISPLAY PREDICTION RESULTS
                            st.markdown(
                                "<div class='stat-card' style='background: #1e3a8a;"
                                "border-color: #3b82f6;'>"
                                "<div class='label' style='color: #bae6fd;'>Prediction</div>"
                                f"<div class='value' style='font-size: 2.2rem;'>{pred_class}</div>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                            ### DISPLAY PREDICTION DISCRIMINATIVE FEATURES
                            st.markdown(
                                section_label("Top Contributing Features"), unsafe_allow_html=True
                            )
                            contribs = X_test_scaled[0] * clf.coef_[0]
                            contrib_data = []

                            ### GET FEATURES
                            for name, scaled_val, coef, contrib in zip(
                                feature_names, X_test_scaled[0], clf.coef_[0], contribs
                            ):
                                if coef != 0:
                                    direction = "Pushed toward Fastball"\
                                        if contrib > 0 else "Pushed toward Offspeed"
                                    contrib_data.append({
                                        "Feature": name,
                                        "Standardized Size": scaled_val,
                                        "LogReg β Weight": coef,
                                        "Net Contribution": contrib,
                                        "Effect": direction,
                                    })

                            ### DISPLAY FEATURES
                            if contrib_data:
                                contrib_df = pd.DataFrame(contrib_data).sort_values(
                                    by="Net Contribution", key=abs, ascending=False
                                ).head(8)
                                st.dataframe(
                                    contrib_df.style.format({
                                        "Standardized Size": "{:+.2f}",
                                        "LogReg β Weight": "{:+.4f}",
                                        "Net Contribution": "{:+.4f}",
                                    })
                                    .background_gradient(
                                        subset=["Net Contribution"], cmap="RdBu_r"
                                    )
                                    .set_properties(
                                        **{"font-family": "monospace"},
                                        subset=["Feature", "Net Contribution"]
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
                    except Exception as e:
                        st.error(f"Error analyzing kinematics: {e}")

