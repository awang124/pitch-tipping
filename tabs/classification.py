import os
import html
import pandas as pd
import streamlit as st
import processing.tab3_helper as t3

from ui.components import (
    info_panel,
    warn_panel,
    success_panel,
    section_label,
    metric_card,
)
from utils import lr_coefficients_styler

def render_classification_tab(base_dir):
    """
    Render Tab 3 (Pitch Classification).

    Provides Mann-Whitney U-Testing, and Logistic Regression and
    Decision Tree modeling to identify discriminative features.
    Available in single-pitcher or inter-pitcher evaluation modes.
    Stores trained single-pitcher models in session state for Tab 4.

    Parameters
    ----------
    base_dir: str
        Root directory for Pitches/ and Keypoints/ folders.
    """

    ### TOP INSTRUCTION PANEL
    st.markdown(
        info_panel(
            "We present three separate approaches for identifying features"
            " significant in discriminating pitch class as cues: <br>"
            "• <b>Decision Tree</b>: features most decreasing overall impurity"
            " after partitions are most indicative of pitch class. <br>"
            "• <b>Logistic Regression</b>: largest weights in magnitude"
            " correspond to features most indicative of pitch class.<br>"
            "• <b>Mann-Whitney U-Test</b>: identifies features whose values least"
            " overlap (best separate) between fastball and off-speed pitches.<br>"
        ),
        unsafe_allow_html=True,
    )

    ### KEYPOINTS DIRECTORY (JUST USE DEFAULT)
    keypoints_root = os.path.join(base_dir, "Keypoints")
    if not os.path.exists(keypoints_root):
        st.markdown(
            warn_panel(f"Keypoints directory not found at `{keypoints_root}`. Run Tab 1 first."),
            unsafe_allow_html=True,
        )
        return

    ### SELECT SINGLE OR INTER-PITCHER
    with st.spinner("Training (this may take a few minutes):"):
        eval_mode = st.radio(
            "Evaluation Type",
            ["Single-Pitcher (80/20 Split)", "Inter-Pitcher (Train on John, Test on James)"],
            horizontal=True,
        )

        if eval_mode == "Inter-Pitcher (Train on John, Test on James)":
            ### BOXES TO CHOOSE TWO PITCHERS (THEIR DIRECTORIES)
            c1, c2 = st.columns(2)
            with c1:
                train_root_rel = st.text_input("Training Keypoints Directory", "Keypoints/Kieran")
            with c2:
                test_root_rel = st.text_input("Evaluation Keypoints Directory", "Keypoints/Randy")
            train_root = os.path.join(base_dir, train_root_rel)
            test_root = os.path.join(base_dir, test_root_rel)

            ### COMPUTE KEYPOINT FEATURE REPRESENTATIONS
            raw_seqs_train, n_frames, pf_feature_names = t3.loadFrameDataset(train_root)
            raw_seqs_test, _, _ = t3.loadFrameDataset(test_root)
        else:
            ### BOXES TO CHOOSE SINGLE PITCHER (THEIR DIRECTORY)
            data_root_rel = st.text_input("Keypoints Directory", "Keypoints/Randy")
            data_root = os.path.join(base_dir, data_root_rel)

            ### COMPUTE KEYPOINT FEATURE REPRESENTATIONS
            all_seqs, n_frames, pf_feature_names = t3.loadFrameDataset(data_root)
            raw_seqs_train = all_seqs
            raw_seqs_test = []

        ### INTER-PITCHER
        if eval_mode == "Inter-Pitcher (Train on John, Test on James)":
            if len(raw_seqs_train) < 3 or len(raw_seqs_test) < 1:
                st.warning(
                    "Not enough labeled sequences found. Train:"
                    f" {len(raw_seqs_train)}, Test: {len(raw_seqs_test)}."
                )
                return

            ### FIT DECISION TREE
            n_fastball = sum(1 for _, l in raw_seqs_train if l == 1)
            n_other = len(raw_seqs_train) - n_fastball
            st.markdown(section_label("Decision Tree"), unsafe_allow_html=True)
            dt_results = t3.evaluate_cross_session(
                raw_seqs_train, raw_seqs_test, pf_feature_names,
                model_type="tree", max_depth=10, top_n=10
            )

            ### DISPLAY DECISION TREE METRICS
            st.markdown(
                f"**Augmented Training Set Size:**"
                " {dt_results['augmented_train_size']} sequences"
            )
            st.markdown(f"**Test Set Size:** {dt_results['test_size']} sequences")
            col_acc, col_rep = st.columns([1, 2])
            with col_acc:
                st.metric("Test Accuracy", f"{dt_results['accuracy']:.3f}")
            with col_rep:
                st.text("Test Report:\n" + dt_results['classification_report'])

            ### DISPLAY DECISION TREE DISCRIMINATIVE FEATURES
            st.markdown("**Top 10 Most Discriminative Features:**")
            for feat, imp in dt_results['top_features']:
                st.write(f"- `{feat}` ({imp:.4f})")

            ### FIT LOGISTIC REGRESSION
            st.markdown(section_label("Logistic Regression"), unsafe_allow_html=True)
            agg_results = t3.evaluate_cross_session_aggregate_lr(
                raw_seqs_train, raw_seqs_test, C=1.0, top_n=10
            )

            ### DISPLAY LOGISTIC REGRESSION METRICS
            lr_m1, lr_m2, lr_m3, lr_m4 = st.columns(4)
            with lr_m1:
                st.metric("Test Accuracy", f"{agg_results['accuracy']:.3f}")
            with lr_m2:
                st.metric(
                    "# Features Retained",
                    f"{agg_results['n_nonzero_features']} / {agg_results['total_features']}"
            )
            with lr_m3:
                st.metric(
                    "Augmented Training Set Size", f"{agg_results['augmented_train_size']}"
                )
            with lr_m4:
                st.metric("Test Set Size", f"{agg_results['test_size']}")
            lr_coef_col, lr_rep_col = st.columns([1.2, 1])

            ### DISPLAY LOGISTIC REGRESSION DISCRIMINATIVE FEATURES
            with lr_coef_col:
                st.markdown(
                    "<div class='lr-table-title'>Top 10 Most Discriminative Features</div>",
                    unsafe_allow_html=True,
                )
                _n_lr = len(agg_results["top_features"])
                st.dataframe(
                    lr_coefficients_styler(agg_results["top_features"]),
                    hide_index=True,
                    width="stretch",
                    height=min(460, 48 + _n_lr * 38),
                )
            with lr_rep_col:
                st.markdown("<div style='height:1.35rem;'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='stat-card' style='margin-bottom:0;'>"
                    "<div class='label'>Test Report</div><pre class='model-report-pre'>"
                    f"{html.escape(agg_results['classification_report'])}"
                    "</pre></div>",
                    unsafe_allow_html=True,
                )

        ### SINGLE-PITCHER
        else:
            if len(all_seqs) < 5:
                st.warning(
                    "Not enough sequences to split for Single-Pitcher"
                    f" evaluation. Found {len(all_seqs)}."
                )
                return

            ### FIT DECISION TREE
            n_fastball = sum(1 for _, l in all_seqs if l == 1)
            n_other = len(all_seqs) - n_fastball
            st.markdown(section_label("Decision Tree"), unsafe_allow_html=True)
            dt_results = t3.evaluate_single_session(
                all_seqs, pf_feature_names, model_type="tree", top_n=10
            )

            ### DISPLAY DECISION TREE METRICS
            if dt_results:
                st.markdown(
                    "**Augmented Training Set Size:**"
                    f" {dt_results['augmented_train_size']} sequences"
                )
                st.markdown(f"**Test Set Size:** {dt_results['test_size']} sequences")
                c_acc1, c_rep1 = st.columns([1, 2])
                with c_acc1:
                    st.metric("Test Accuracy", f"{dt_results['accuracy']:.3f}")
                with c_rep1:
                    st.text("Test Report:\n" + dt_results['classification_report'])

            ### FIT LOGISTIC REGRESSION
            st.markdown(section_label("Logistic Regression"), unsafe_allow_html=True)
            agg_results = t3.evaluate_single_session_aggregate_lr(
                all_seqs, test_size=0.2, C=1.0, top_n=10
            )
            if not agg_results:
                st.error(
                    "Error evaluating single session: Not enough samples"
                    " per class to perform a stratified split.")

            else:
                ### DISPLAY LOGISTIC REGRESSION METRICS
                st.session_state["single_pitcher_clf"] = agg_results["clf"]
                st.session_state["single_pitcher_scaler"] = agg_results["scaler"]
                st.session_state["single_pitcher_features"] = agg_results["feature_names"]

                sp_m1, sp_m2, sp_m3, sp_m4 = st.columns(4)
                with sp_m1:
                    st.metric("Test Accuracy", f"{agg_results['accuracy']:.3f}")
                with sp_m2:
                    st.metric(
                        "Features Retained",
                        f"{agg_results['n_nonzero_features']} / {agg_results['total_features']}"
                    )
                with sp_m3:
                    st.metric(
                        "Augmented Training Set Size", f"{agg_results['augmented_train_size']}"
                    )
                with sp_m4:
                    st.metric("Test Set Size", f"{agg_results['test_size']}")

                ### DISPLAY LOGISTIC REGRESSION DISCRIMINATIVE FEATURES
                sp_lr_tab, sp_lr_rep = st.columns([1.2, 1])
                with sp_lr_tab:
                    st.markdown(
                        "<div class='lr-table-title'>Top 10 Most Discriminative Features:</div>",
                        unsafe_allow_html=True,
                    )
                    _n_splr = len(agg_results["top_features"])
                    st.dataframe(
                        lr_coefficients_styler(agg_results["top_features"]),
                        hide_index=True,
                        width="stretch",
                        height=min(460, 48 + _n_splr * 38),
                    )
                with sp_lr_rep:
                    st.markdown("<div style='height:1.35rem;'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='stat-card' style='margin-bottom:0;'>"
                        "<div class='label'>Test Report</div><pre class='model-report-pre'>"
                        f"{html.escape(agg_results['classification_report'])}</pre>"
                        "</pre></div>",
                        unsafe_allow_html=True,
                    )

    ### PERFORM MANN-WHITNEY U-TEST
    st.markdown(section_label("Mann-Whitney U-Test"), unsafe_allow_html=True)
    seqs = t3.loadSequences(keypoints_root)
    relabel = dict.fromkeys(
        ["cutter", "fastball", "four_seam_fastball", "two_seam_fastball"], "fastball"
    )
    relabel.update(dict.fromkeys(
        ["changeup", "curveball", "offspeed", "slider", "splitter"], "offspeed")
    )
    seqs = [(pair[0], relabel[pair[1]]) for pair in seqs if pair[1] in relabel]

    if len(seqs) < 4:
        st.markdown(
            warn_panel("Not enough labeled sequences for statistical testing (need >= 4)."),
            unsafe_allow_html=True
        )
    else:
        raw_features = [t3.extractFeatures(seq, label) for seq, label in seqs]
        raw_features = [f for f in raw_features if len(f) > 1]
        if len(raw_features) < 4:
            st.markdown(warn_panel(
                "Not enough sequences with extractable features for statistical testing."),
                unsafe_allow_html=True
            )
        else:
            features_df = pd.DataFrame(raw_features)
            mann_table = t3.mannWhitney(features_df, n_show=15)
            st.dataframe(
                mann_table.style.format({"p-value": "{:.4f}", "Effect Size": "{:.3f}"}),
                hide_index=True,
                width="stretch",
                height=400,
            )

