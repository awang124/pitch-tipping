CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root tokens ────────────────────────────────────────────────────── */
:root {
    --bg:         #0a0d12;
    --surface:    #111620;
    --surface2:   #181e2c;
    --border:     #1e2a3a;
    --accent:     #2563eb;
    --accent2:    #38bdf8;
    --warn:       #f59e0b;
    --success:    #22c55e;
    --danger:     #ef4444;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --mono:       'IBM Plex Mono', monospace;
    --sans:       'IBM Plex Sans', sans-serif;
}

/* ── Global reset ───────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ──────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 0.6rem !important;
    font-weight: 600 !important;
}

/* ── Top wordmark banner ────────────────────────────────────────────── */
.wordmark {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 0.2rem;
}
.wordmark-primary {
    font-family: var(--mono);
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.wordmark-badge {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    color: var(--accent2);
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    padding: 2px 8px;
    border-radius: 2px;
    text-transform: uppercase;
}
.wordmark-sub {
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.02em;
    margin-bottom: 2rem;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 100%);
    margin: 1.5rem 0 2rem;
    opacity: 0.4;
}

/* ── Section labels ─────────────────────────────────────────────────── */
.section-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 16px;
    height: 1px;
    background: var(--accent);
    opacity: 0.7;
}

/* ── Stat cards ─────────────────────────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.stat-card .label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.stat-card .value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
}
.stat-card .unit {
    font-size: 0.75rem;
    color: var(--muted);
    margin-left: 4px;
}

/* ── Info panel ─────────────────────────────────────────────────────── */
.info-panel {
    background: var(--surface2);
    border-left: 2px solid var(--accent);
    border-radius: 0 4px 4px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.6;
}
.warn-panel {
    background: rgba(245,158,11,0.06);
    border-left: 2px solid var(--warn);
    border-radius: 0 4px 4px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #fcd34d;
    line-height: 1.6;
}
.success-panel {
    background: rgba(34,197,94,0.06);
    border-left: 2px solid var(--success);
    border-radius: 0 4px 4px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
    color: #86efac;
    line-height: 1.6;
}

/* ── ML report / coefficient tables ─────────────────────────────────── */
.model-report-pre {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: #cbd5e1;
    line-height: 1.5;
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
}
.lr-table-title {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 0.5rem 0;
}

/* ── Tabs ───────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    margin-bottom: 2rem !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    padding: 0.6rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Inputs & selects ───────────────────────────────────────────────── */
[data-baseweb="input"] input,
[data-baseweb="select"] div,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
}
label[data-testid="stWidgetLabel"] p,
.stSelectbox label p,
.stTextInput label p {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
}

/* ── File uploader ──────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"] label {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: rgba(37,99,235,0.1) !important;
    color: var(--accent2) !important;
}
.stButton [kind="primary"] > button,
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
}

/* ── Progress bar ───────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}
[data-testid="stProgressBar"] > div {
    background: var(--surface) !important;
    border-radius: 2px !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}

/* ── Columns ─────────────────────────────────────────────────────────── */
[data-testid="column"] {
    padding: 0 0.75rem !important;
}

/* ── Video player ────────────────────────────────────────────────────── */
video {
    border-radius: 4px !important;
    border: 1px solid var(--border) !important;
}

/* ── st.success / st.warning / st.error overrides ───────────────────── */
[data-testid="stAlert"] {
    border-radius: 3px !important;
    font-family: var(--sans) !important;
    font-size: 0.82rem !important;
}
</style>
"""

