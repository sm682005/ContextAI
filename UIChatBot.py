import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Mood Config ─────────────────────────────────────────────────────────────────
MOODS = {
    "Happy 😊": {
        "prompt": "You are a happy AI agent. You respond cheerfully and enthusiastically.",
        "color": "#FFD700",
        "glow": "rgba(255, 215, 0, 0.55)",
        "bg": "rgba(255, 215, 0, 0.07)",
        "emoji": "😊",
        "label": "HAPPY",
    },
    "Sad 😢": {
        "prompt": "You are a very sad AI agent. You respond in a depressed and emotional tone.",
        "color": "#5B9BD5",
        "glow": "rgba(91, 155, 213, 0.55)",
        "bg": "rgba(91, 155, 213, 0.07)",
        "emoji": "😢",
        "label": "SAD",
    },
    "Angry 😠": {
        "prompt": "You are an angry AI agent. You respond aggressively and impatiently.",
        "color": "#FF4444",
        "glow": "rgba(255, 68, 68, 0.55)",
        "bg": "rgba(255, 68, 68, 0.07)",
        "emoji": "😠",
        "label": "ANGRY",
    },
    "Sarcastic 🙄": {
        "prompt": "You are a sarcastic AI agent. You respond with irony and cynicism.",
        "color": "#B47FFF",
        "glow": "rgba(180, 127, 255, 0.55)",
        "bg": "rgba(180, 127, 255, 0.07)",
        "emoji": "🙄",
        "label": "SARCASTIC",
    },
    "Funny 😂": {
        "prompt": "You are a humorous AI agent. You respond with jokes and comedic timing.",
        "color": "#FF8C00",
        "glow": "rgba(255, 140, 0, 0.55)",
        "bg": "rgba(255, 140, 0, 0.07)",
        "emoji": "😂",
        "label": "FUNNY",
    },
    "Mature 🧠": {
        "prompt": "You are a mature AI agent. You respond with wisdom and depth.",
        "color": "#00E5CC",
        "glow": "rgba(0, 229, 204, 0.55)",
        "bg": "rgba(0, 229, 204, 0.07)",
        "emoji": "🧠",
        "label": "MATURE",
    },
}

# ─── Session State Init ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_mood" not in st.session_state:
    st.session_state.selected_mood = None
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

# ─── CSS ─────────────────────────────────────────────────────────────────────────
def get_css(mood_color, mood_glow, mood_bg):
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Inter:wght@300;400;500&display=swap');

:root {{
    --mood-color: {mood_color};
    --mood-glow: {mood_glow};
    --mood-bg: {mood_bg};
    --dark: #0B0D17;
    --card: #111320;
    --surface: #161929;
    --border: rgba(255,255,255,0.06);
    --text: #E4E8F0;
    --muted: #6B7280;
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}}

[data-testid="stSidebar"] {{
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}}

/* Hide default streamlit header */
[data-testid="stHeader"] {{ display: none; }}
footer {{ display: none; }}
#MainMenu {{ display: none; }}

/* ── Custom header ── */
.moodbot-header {{
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--mood-color);
    text-shadow: 0 0 24px var(--mood-glow), 0 0 60px var(--mood-glow);
    letter-spacing: 0.12em;
    text-align: center;
    padding: 1.2rem 0 0.2rem;
    transition: all 0.6s ease;
    animation: headerPulse 3s ease-in-out infinite;
}}

@keyframes headerPulse {{
    0%, 100% {{ text-shadow: 0 0 24px var(--mood-glow), 0 0 60px var(--mood-glow); }}
    50% {{ text-shadow: 0 0 36px var(--mood-glow), 0 0 90px var(--mood-glow), 0 0 120px var(--mood-glow); }}
}}

.moodbot-sub {{
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    font-family: 'Orbitron', monospace;
}}

/* ── Active mood badge ── */
.mood-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--mood-bg);
    border: 1px solid var(--mood-color);
    color: var(--mood-color);
    border-radius: 50px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    letter-spacing: 0.15em;
    box-shadow: 0 0 12px var(--mood-glow);
    margin: 0 auto 1.2rem;
    width: fit-content;
}}

/* ── Chat bubbles ── */
.chat-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 0.5rem 0;
}}

.bubble-row {{
    display: flex;
    animation: slideIn 0.35s cubic-bezier(.22,1,.36,1) forwards;
    opacity: 0;
}}

@keyframes slideIn {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.bubble-row.user {{ justify-content: flex-end; }}
.bubble-row.bot  {{ justify-content: flex-start; }}

.bubble {{
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    border-radius: 18px;
    font-size: 0.93rem;
    line-height: 1.6;
    position: relative;
}}

.bubble.user {{
    background: var(--mood-bg);
    border: 1px solid var(--mood-color);
    border-bottom-right-radius: 4px;
    color: var(--text);
    box-shadow: 0 0 12px var(--mood-glow);
}}

.bubble.bot {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text);
}}

.bubble-label {{
    font-size: 0.65rem;
    font-family: 'Orbitron', monospace;
    letter-spacing: 0.18em;
    font-weight: 600;
    margin-bottom: 0.3rem;
    color: var(--mood-color);
    text-transform: uppercase;
}}

.bubble-label.bot-label {{ color: var(--muted); }}

/* ── Mood card buttons (sidebar) ── */
.mood-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
    cursor: pointer;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 0.88rem;
    color: var(--text);
}}

.mood-card:hover {{
    border-color: var(--mood-color);
    box-shadow: 0 0 14px var(--mood-glow);
    transform: translateX(4px);
}}

.mood-card.active {{
    border-color: var(--mood-color);
    background: var(--mood-bg);
    box-shadow: 0 0 18px var(--mood-glow);
}}

/* ── Input area ── */
[data-testid="stTextInput"] input {{
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.2s ease !important;
}}

[data-testid="stTextInput"] input:focus {{
    border-color: var(--mood-color) !important;
    box-shadow: 0 0 12px var(--mood-glow) !important;
    outline: none !important;
}}

/* ── Send button ── */
[data-testid="stFormSubmitButton"] button,
.stButton > button {{
    background: transparent !important;
    border: 1px solid var(--mood-color) !important;
    color: var(--mood-color) !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    font-size: 0.78rem !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 8px var(--mood-glow) !important;
}}

[data-testid="stFormSubmitButton"] button:hover,
.stButton > button:hover {{
    background: var(--mood-bg) !important;
    box-shadow: 0 0 20px var(--mood-glow) !important;
    transform: translateY(-1px) !important;
}}

/* ── Sidebar label ── */
.sidebar-section-label {{
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: var(--muted);
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem;
    padding-left: 0.1rem;
}}

/* ── Divider ── */
.neon-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--mood-color), transparent);
    border: none;
    margin: 0.8rem 0 1.2rem;
    opacity: 0.5;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 4rem 1rem;
    color: var(--muted);
}}
.empty-state .big-emoji {{ font-size: 3.5rem; margin-bottom: 1rem; }}
.empty-state p {{ font-size: 0.9rem; line-height: 1.7; }}

/* ── Stats strip ── */
.stats-strip {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}}
.stat-chip {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.75rem;
    font-size: 0.72rem;
    color: var(--muted);
    font-family: 'Orbitron', monospace;
    letter-spacing: 0.1em;
}}

/* ── Spinner override ── */
[data-testid="stSpinner"] {{ color: var(--mood-color) !important; }}

/* ── Model loading card ── */
.loading-card {{
    background: var(--card);
    border: 1px solid var(--mood-color);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 24px var(--mood-glow);
    font-family: 'Orbitron', monospace;
    color: var(--mood-color);
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    margin: 2rem 0;
}}

/* Scrollable chat area */
.chat-scroll-area {{
    max-height: 58vh;
    overflow-y: auto;
    padding-right: 6px;
    scrollbar-width: thin;
    scrollbar-color: var(--mood-color) transparent;
}}
.chat-scroll-area::-webkit-scrollbar {{ width: 4px; }}
.chat-scroll-area::-webkit-scrollbar-thumb {{ background: var(--mood-color); border-radius: 4px; }}
</style>
"""

# ─── Model Loader ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "temperature": 0.9,
            "max_new_tokens": 256,
            "do_sample": True,
            "repetition_penalty": 1.03,
        },
    )
    return ChatHuggingFace(llm=llm)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    mood = st.session_state.selected_mood
    mc = MOODS[mood] if mood else {"color": "#00E5CC", "glow": "rgba(0,229,204,0.4)", "bg": "rgba(0,229,204,0.07)"}

    st.markdown(get_css(mc["color"], mc["glow"], mc["bg"]), unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">⚡ Select AI Mood</div>', unsafe_allow_html=True)

    for mood_name, mood_data in MOODS.items():
        is_active = st.session_state.selected_mood == mood_name
        active_class = "active" if is_active else ""
        if st.button(
            f"{mood_data['emoji']}  {mood_name}",
            key=f"mood_{mood_name}",
            use_container_width=True,
        ):
            if st.session_state.selected_mood != mood_name:
                st.session_state.selected_mood = mood_name
                st.session_state.messages = [SystemMessage(content=mood_data["prompt"])]
                if st.session_state.chat_model is None:
                    with st.spinner("Booting neural core…"):
                        st.session_state.chat_model = load_model()
                st.rerun()

    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-label">🗂 Session</div>', unsafe_allow_html=True)

    msg_count = sum(1 for m in st.session_state.messages if isinstance(m, HumanMessage))
    st.markdown(f'<div class="stat-chip">💬 {msg_count} messages</div>', unsafe_allow_html=True)

    if st.button("🗑 Clear Chat", use_container_width=True):
        if st.session_state.selected_mood:
            mood_data = MOODS[st.session_state.selected_mood]
            st.session_state.messages = [SystemMessage(content=mood_data["prompt"])]
            st.rerun()

# ─── Main Area ───────────────────────────────────────────────────────────────────
mood = st.session_state.selected_mood
mc = MOODS[mood] if mood else {"color": "#00E5CC", "glow": "rgba(0,229,204,0.4)", "bg": "rgba(0,229,204,0.07)", "label": "NONE", "emoji": "🤖"}

st.markdown(get_css(mc["color"], mc["glow"], mc["bg"]), unsafe_allow_html=True)

st.markdown('<div class="moodbot-header">MOODBOT AI</div>', unsafe_allow_html=True)
st.markdown('<div class="moodbot-sub">Neural Language Interface · v1.0</div>', unsafe_allow_html=True)

if mood:
    st.markdown(
        f'<div style="text-align:center"><div class="mood-badge">● {mc["label"]} MODE ACTIVE {mc["emoji"]}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ── Chat Display ─────────────────────────────────────────────────────────────────
chat_messages = [m for m in st.session_state.messages if not isinstance(m, SystemMessage)]

if not mood:
    st.markdown("""
    <div class="empty-state">
        <div class="big-emoji">🤖</div>
        <p>Select a <strong>mood</strong> from the sidebar<br>to activate your AI companion.</p>
    </div>
    """, unsafe_allow_html=True)
elif not chat_messages:
    st.markdown(f"""
    <div class="empty-state">
        <div class="big-emoji">{mc['emoji']}</div>
        <p>MoodBot is in <strong>{mc['label']}</strong> mode.<br>Say something to begin the conversation.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    bubbles_html = '<div class="chat-scroll-area"><div class="chat-wrapper">'
    for i, msg in enumerate(chat_messages):
        delay = min(i * 0.05, 0.5)
        if isinstance(msg, HumanMessage):
            bubbles_html += f"""
            <div class="bubble-row user" style="animation-delay:{delay}s">
                <div>
                    <div class="bubble-label" style="text-align:right">YOU</div>
                    <div class="bubble user">{msg.content}</div>
                </div>
            </div>"""
        elif isinstance(msg, AIMessage):
            bubbles_html += f"""
            <div class="bubble-row bot" style="animation-delay:{delay}s">
                <div>
                    <div class="bubble-label bot-label">MOODBOT {mc['emoji']}</div>
                    <div class="bubble bot">{msg.content}</div>
                </div>
            </div>"""
    bubbles_html += '</div></div>'
    st.markdown(bubbles_html, unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────────────────────────
if mood:
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                label="message",
                placeholder=f"Talk to MoodBot in {mc['label']} mode…",
                label_visibility="collapsed",
            )
        with col2:
            submit = st.form_submit_button("SEND ▶", use_container_width=True)

    if submit and user_input.strip():
        st.session_state.messages.append(HumanMessage(content=user_input.strip()))

        if st.session_state.chat_model is None:
            with st.spinner("Loading neural core…"):
                st.session_state.chat_model = load_model()

        with st.spinner("MoodBot is thinking…"):
            response = st.session_state.chat_model.invoke(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))

        st.rerun()