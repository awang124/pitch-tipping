import streamlit as st
import ui.style as style
import ui.sidebar as sidebar

from tabs.compare import render_compare_tab
from tabs.estimation import render_estimation_tab
from tabs.visualization import render_visualization_tab
from tabs.classification import render_classification_tab
from tabs.prediction import render_prediction_tab
from tabs.overlay import render_overlay_tab

### TAB NAME
st.set_page_config(
    page_title="PitchVision",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(style.CSS, unsafe_allow_html=True)

### CREATE SIDEBAR
with st.sidebar:
    base_dir = sidebar.render()

### CREATE TOP TITLE PANEL
st.markdown(
    """
    <div class='wordmark'><span class='wordmark-primary'>PitchVision</span></div>
    <div class='wordmark-sub'>
        Automated Pitch Tipping Detection via Motion Estimation and Statistical Modeling
    </div><div class='divider'></div>
    """,
    unsafe_allow_html=True,
)

### CREATE TABS
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pitch Compare",
    "1: Motion Estimation",
    "2: Comparative Visualization",
    "3: Pitch Classification",
    "4: Pitch Prediction",
    "5: Skeleton Overlay",
])

with tab0:
    render_compare_tab(base_dir)

with tab1:
    render_estimation_tab(base_dir)

with tab2:
    render_visualization_tab(base_dir)

with tab3:
    render_classification_tab(base_dir)

with tab4:
    render_prediction_tab(base_dir)

with tab5:
    render_overlay_tab(base_dir)
