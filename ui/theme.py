"""Shared dashboard colours and Streamlit styling."""

TEMPERATURE_RAMP = ["#1d2a73", "#2f6fe4", "#38bdf8", "#4ade80", "#facc15", "#fb923c", "#ef4444"]

CONTROLLER_COLOURS = {
    "Bound: max cooling": "#b9c4d4",
    "Bound: min cooling": "#8895a8",
    "Fixed setpoint": "#94a3b8",
    "Rule-based (no hysteresis)": "#34d399",
    "Rule-based": "#38bdf8",
    "Q-Learning": "#a78bfa",
    "SARSA": "#f59e0b",
    "DQN": "#34d399",
    "PPO": "#fb7185",
}


def apply_control_room_theme(st) -> None:
    """Apply restrained control-room styling without changing Streamlit functionality."""
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #0e1a2f; }
        [data-testid="stSidebar"] { background: #0e1a2f; border-right: 1px solid #274068; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .control-kicker { color: #38bdf8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; }
        .control-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem; }
        .safety-banner { background: #1b2d52; border-left: 4px solid #38bdf8; border-radius: 0.4rem; padding: 0.4rem 0.8rem; margin: 0.75rem 0 1.2rem; }
        .safety-chip { display: inline-block; padding: 0.28rem 0.65rem; border-radius: 999px; font-weight: 700; font-size: 0.82rem; }
        .safety-green { background: #064e3b; color: #a7f3d0; }
        .safety-amber { background: #78350f; color: #fde68a; }
        .safety-red { background: #7f1d1d; color: #fecaca; }
        div[data-testid="stMetric"] { background: #142340; border: 1px solid #274068; border-radius: 0.5rem; padding: 0.8rem; }
        [class*="st-key-panel_"], [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stLayoutWrapper"]:has(> [data-testid="stVerticalBlock"][data-border="true"]) {background:#142340;border-color:#274068;border-radius:12px;}
        [data-testid="stHorizontalBlock"]:has([class*="st-key-panel_"]) {align-items:stretch;}
        [data-testid="stColumn"] > [data-testid="stVerticalBlock"]:has([class*="st-key-panel_"]), [data-testid="stColumn"] [class*="st-key-panel_"] {height:100%;}
        [data-testid="stAppViewContainer"] {font-family:Inter,"Segoe UI",system-ui,sans-serif;}
        [role="tab"], [class*="st-key-panel_"] strong {text-transform:uppercase;letter-spacing:.1em;font-size:11px;}
        [role="tab"][aria-selected="true"] {color:#38bdf8;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(st, title, caption):
    panel = st.container(
        border=True, height="stretch", key="panel_" + title.lower().replace(" ", "_")
    )
    panel.markdown(f"**{title}**")
    panel.caption(caption)
    return panel
