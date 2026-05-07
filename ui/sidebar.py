import streamlit as st

def render():
    """
    Render dashboard sidebar with base directory configuration and pipeline modules.

    Returns
    -------
    User-specified base directory.
    """

    ### SIDEBAR TOP TITLE PANEL
    st.markdown(
        """
        <div style='padding: 1.2rem 0 0.5rem;'>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 1.05rem;
        font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em;'>PitchVision</div>
        <div style='font-size: 0.68rem; color: #64748b; margin-top: 2px;
        letter-spacing: 0.04em;'>Sports Analytics Research Group</div>
        <div style='font-size: 0.68rem; color: #475569; letter-spacing: 0.04em;'>
        Johns Hopkins University</div>
        </div>
        <div style='height: 1px; background: #1e2a3a; margin: 1rem 0 1.5rem;'></div>
        """,
        unsafe_allow_html=True,
    )

    ### BASE DIRECTORY INPUT
    st.markdown("### Configuration")
    base_dir = st.text_input(
        "Base Directory", ".", help="Base directory for Pitches/ and Keypoints/."
    )

    ### PIPELINE MODULE LIST
    st.markdown(
        """
        <div style='height: 1px; background: #1e2a3a; margin: 1.2rem 0;'></div>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.58rem;
        letter-spacing: 0.14em; text-transform: uppercase; color: #475569;
        margin-bottom: 0.6rem;'>Pipeline Modules</div>
        """,
        unsafe_allow_html=True,
    )

    pipeline_items = [
        ("Motion Estimation", "Pose & Glove Tracking"),
        ("Feature Engineering", "Aggregate & Per-Frame Cues"),
        ("Cue Identification", "Statistical Testing & ML Modeling"),
        ("Visualization", "DTW & RANSAC Aligned Superimposition"),
    ]
    for name, desc in pipeline_items:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:8px; padding: 5px 0;'>
            <div style='width:6px; height:6px; border-radius:50%;
            background:#22c55e; flex-shrink:0;'></div>
            <div><div style='font-size:0.75rem; color:#cbd5e1;
            font-family:"IBM Plex Mono",monospace;'>{name}</div>
            <div style='font-size:0.65rem; color:#475569;'>{desc}</div>
            </div></div>
            """,
            unsafe_allow_html=True,
        )

    return base_dir

