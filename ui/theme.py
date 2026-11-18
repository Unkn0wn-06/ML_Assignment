"""Shared dashboard colours and Streamlit styling."""

CONTROLLER_COLOURS = {
    "Fixed setpoint": "#94a3b8",
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
        [data-testid="stAppViewContainer"] { background: #07111f; }
        [data-testid="stSidebar"] { background: #0b1728; border-right: 1px solid #1e3a5f; }
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .control-kicker { color: #38bdf8; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; }
        .control-title { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem; }
        .safety-banner { background: #172554; border-left: 4px solid #38bdf8; border-radius: 0.4rem; padding: 0.8rem 1rem; margin: 0.75rem 0 1.2rem; }
        .safety-chip { display: inline-block; padding: 0.28rem 0.65rem; border-radius: 999px; font-weight: 700; font-size: 0.82rem; }
        .safety-green { background: #064e3b; color: #a7f3d0; }
        .safety-amber { background: #78350f; color: #fde68a; }
        .safety-red { background: #7f1d1d; color: #fecaca; }
        div[data-testid="stMetric"] { background: #0f1d2f; border: 1px solid #1e3a5f; border-radius: 0.5rem; padding: 0.8rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
