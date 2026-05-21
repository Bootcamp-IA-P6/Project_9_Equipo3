"""
src/app/streamlit_app.py

App SignalMod — detección de hate speech estilo YouTube.
Ejecutar: streamlit run src/app/streamlit_app.py
"""

import html
import sys
import random
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

from transformers.utils import logging
logging.set_verbosity_error()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.service.model_service import ModelService, AVAILABLE_MODELS
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from service.model_service import ModelService, AVAILABLE_MODELS

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalMod",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Nota: NO ocultamos el header completo para preservar el botón de toggle del sidebar.
# Solo ocultamos el menú hamburguesa y el footer de Streamlit.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=YouTube+Sans:wght@400;600;700&display=swap');

/* ── Ocultar solo elementos de branding, NO el header completo ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Fondo de la app: blanco limpio ── */
.stApp { background: #ffffff; }

/* ── Sidebar oscuro (como YouTube) ── */
section[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #0f0f0f !important;
}
/* Texto del sidebar en blanco */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
}
/* Botones del sidebar */
section[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    color: #e0e0e0 !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 10px !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
}
/* Botón activo en el sidebar */
section[data-testid="stSidebar"] .stButton button[data-active="true"] {
    background: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Divider del sidebar */
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
/* Badge de modelo activo en sidebar */
.sidebar-model-info {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 0.75rem;
    color: #aaaaaa;
}
.sidebar-model-info strong { color: #ffffff; }

/* ── Área principal: fondo blanco, texto oscuro ── */
.main-area { background: #ffffff; }

/* ── Video thumbnail ── */
.video-thumb {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a0a2e 50%, #0d1a1a 100%);
    border-radius: 12px;
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.play-btn {
    width: 72px; height: 72px;
    background: rgba(255,255,255,0.9);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}

/* ── Títulos de video ── */
.video-title {
    font-size: 1.15rem; font-weight: 700;
    color: #0f0f0f; margin: 0.75rem 0 0.3rem;
    line-height: 1.4;
}
.video-meta { font-size: 0.82rem; color: #606060; }
.channel-name { font-weight: 600; font-size: 0.9rem; color: #0f0f0f; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 2px 9px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 700;
    margin-left: 6px; vertical-align: middle;
}
.badge-toxic { background: #cc0000; color: #ffffff; }
.badge-safe  { background: #00c853; color: #ffffff; }

/* ── Comentarios ── */
.comment-wrap {
    display: flex; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid #f0f0f0;
}
.c-avatar {
    width: 36px; height: 36px; min-width: 36px;
    border-radius: 50%; background: #cc0000;
    display: flex; align-items: center; justify-content: center;
    color: #ffffff; font-weight: 700; font-size: 0.85rem;
    flex-shrink: 0;
}
.c-avatar.safe { background: #606060; }
.c-body { flex: 1; min-width: 0; }
.c-header { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; }
.c-user { font-size: 0.84rem; font-weight: 600; color: #0f0f0f; }
.c-time { font-size: 0.75rem; color: #909090; margin-left: 4px; }
.c-text { font-size: 0.88rem; color: #2d2d2d; margin-top: 4px; line-height: 1.55; }
.c-text.toxic {
    background: #fff5f5;
    border-left: 3px solid #cc0000;
    padding: 6px 10px; border-radius: 0 6px 6px 0;
    margin-top: 6px;
}
.c-flagged { font-size: 0.77rem; color: #cc0000; font-weight: 500; margin-top: 4px; }

/* ── Toxicity bar inline ── */
.tox-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.8rem; color: #606060; margin-top: 6px; flex-wrap: wrap;
}
.tox-bar-bg {
    flex: 1; max-width: 120px;
    background: #e5e5e5; border-radius: 4px; height: 6px;
}
.tox-bar-fill { height: 6px; border-radius: 4px; }

/* ── Sugeridos ── */
.sug-card {
    display: flex; gap: 8px; margin-bottom: 10px;
    cursor: pointer;
}
.sug-thumb {
    width: 120px; min-width: 120px; height: 68px;
    background: #1a1a2e; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
}
.sug-title  { font-size: 0.82rem; font-weight: 600; color: #0f0f0f; line-height: 1.3; }
.sug-ch     { font-size: 0.75rem; color: #606060; margin-top: 2px; }
.sug-meta   { font-size: 0.72rem; color: #909090; }

/* ── Section header ── */
.sec-title {
    font-size: 1rem; font-weight: 700; color: #0f0f0f;
    margin: 1.25rem 0 0.75rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e5e5;
}

/* ── Modal body fixes ── */
[data-testid="stDialog"] { background: #ffffff; }

/* ── Hub cards ── */
.hub-card {
    background: #ffffff; border: 1px solid #e5e5e5;
    border-radius: 12px; padding: 1rem;
}
.hub-kpi-label { font-size: 0.72rem; color: #606060; text-transform: uppercase;
                  letter-spacing: 0.5px; margin-bottom: 4px; }
.hub-kpi-val   { font-size: 1.8rem; font-weight: 700; color: #0f0f0f; }

/* ── Model cards (settings) ── */
.model-card {
    background: #ffffff; border: 1.5px solid #e5e5e5;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}
.model-card.active {
    border-color: #cc0000; background: #fff5f5;
}
.model-card-name { font-size: 0.95rem; font-weight: 600; color: #0f0f0f; }
.model-card-desc { font-size: 0.8rem; color: #606060; margin-top: 3px; }
.model-pill {
    display: inline-block; background: #f0f0f0; color: #333;
    border-radius: 6px; padding: 2px 8px; font-size: 0.73rem; margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "page"          : "Home",
        "selected_model": list(AVAILABLE_MODELS.keys())[0],
        "threshold"     : 0.5,
        "pending_modal" : None,   # dict con el comentario pendiente de decisión
        "comments": [
            {"user": "user_prime", "initial": "U",
             "text": "Excelente video, muy informativo!", "time": "1 h",
             "is_toxic": False, "probability": 0.04, "labels": []},
            {"user": "troll_master", "initial": "T",
             "text": "Esto es una basura completa", "time": "30 min",
             "is_toxic": True, "probability": 0.91, "labels": ["Insulto","Agresividad"]},
            {"user": "curious_viewer", "initial": "C",
             "text": "¿Alguien puede explicar esto mejor?", "time": "15 min",
             "is_toxic": False, "probability": 0.07, "labels": []},
        ],
        "hub_history": [
            {"Usuario": "@user_992",   "Comentario": '"No puedo creer que seas tan..."', "Score": 0.94, "Acción": "🚫 Bloqueado"},
            {"Usuario": "@alpha_mod",  "Comentario": '"Spam repetitivo de enlaces."',    "Score": 0.82, "Acción": "🚩 Revisión"},
            {"Usuario": "@anon_404",   "Comentario": '"Discurso de odio en contexto."',  "Score": 0.98, "Acción": "📋 Archivado"},
            {"Usuario": "@user_123",   "Comentario": '"¡Gran contenido, sigan!"',        "Score": 0.03, "Acción": "✅ Aprobado"},
            {"Usuario": "@viewer_x",   "Comentario": '"Esta gente debería desaparecer."',"Score": 0.97, "Acción": "🚫 Bloqueado"},
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Model cache ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo...")
def get_service(model_name: str) -> ModelService:
    return ModelService(model_name, PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            "<div style='padding:0.5rem 0 0.25rem; font-size:1.3rem; font-weight:700;'>"
            "🎬 <span style='color:#cc0000'>Signal</span>Mod</div>"
            "<div style='font-size:0.65rem; color:#aaa; margin-bottom:1.2rem;'>"
            "Signal within the Noise</div>",
            unsafe_allow_html=True,
        )

        nav = {"Home": "🏠", "Moderator Hub": "📊", "Settings": "⚙️"}
        for page, icon in nav.items():
            label = f"{icon}  {page}"
            clicked = st.button(label, key=f"nav_{page}", use_container_width=True)
            if clicked:
                st.session_state.page = page
                st.rerun()

        st.divider()

        # Info modelo activo
        model_short = st.session_state.selected_model.split("(")[0].strip()
        tox_cnt     = sum(1 for c in st.session_state.comments if c["is_toxic"])
        total_c     = len(st.session_state.comments)

        st.markdown(
            f"<div class='sidebar-model-info'>"
            f"Modelo activo<br><strong>{html.escape(model_short)}</strong>"
            f"<br><br>Comentarios: <strong>{total_c}</strong>"
            f" · Tóxicos: <strong style='color:#cc0000'>{tox_cnt}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODAL — toxicidad detectada
# ══════════════════════════════════════════════════════════════════════════════
@st.dialog("⚠️  Aviso de Toxicidad Detectada")
def show_toxicity_modal():
    """
    @st.dialog crea una ventana modal nativa de Streamlit (1.32+).
    Cuando se llama a la función decorada, Streamlit renderiza el contenido
    dentro de un overlay modal y pausa la ejecución normal del script.
    """
    data  = st.session_state.pending_modal
    if not data:
        st.rerun()
        return

    text  = data["text"]
    prob  = data["probability"]
    lbls  = data["labels"]
    pct   = int(prob * 100)
    color = "#cc0000" if pct >= 70 else "#ff6d00" if pct >= 40 else "#f5a623"

    st.markdown(
        "<div style='text-align:center; font-size:3rem; color:#cc0000'>⚠️</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:#f8f8f8; border-radius:8px; padding:12px 16px;"
        f"font-style:italic; color:#333; text-align:center; margin:8px 0;'>"
        f"&quot;{html.escape(text[:140])}{'...' if len(text)>140 else ''}&quot;</div>",
        unsafe_allow_html=True,
    )

    # Barra de toxicidad
    st.markdown(
        f"<div style='display:flex; justify-content:space-between; "
        f"font-size:0.82rem; color:#606060; margin-top:12px;'>"
        f"<span>ÍNDICE DE TOXICIDAD</span>"
        f"<span style='color:{color}; font-weight:700'>{pct}%</span></div>"
        f"<div style='background:#e5e5e5; border-radius:4px; height:8px; margin-top:4px;'>"
        f"<div style='width:{pct}%; background:{color}; height:8px; border-radius:4px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Etiquetas
    if lbls:
        tags = " ".join(
            f"<span style='background:#ffe5e5; color:#cc0000; border-radius:14px;"
            f"padding:3px 10px; font-size:0.76rem; font-weight:600; margin:3px;'>"
            f"🚩 {html.escape(l)}</span>"
            for l in lbls
        )
        st.markdown(f"<div style='margin-top:10px'>{tags}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✏️  Editar comentario", use_container_width=True, type="primary"):
            st.session_state.pending_modal = None
            st.rerun()
    with col2:
        if st.button("Publicar de todas maneras", use_container_width=True):
            # Publicar aunque sea tóxico
            c = st.session_state.pending_modal
            st.session_state.comments.append(c)
            st.session_state.hub_history.insert(0, {
                "Usuario"  : "@usuario",
                "Comentario": f'"{c["text"][:45]}..."',
                "Score"    : round(c["probability"], 2),
                "Acción"   : "⚠️ Override usuario",
            })
            st.session_state.pending_modal = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HOME — interfaz estilo YouTube
# ══════════════════════════════════════════════════════════════════════════════
def render_home():
    # Disparar modal si hay comentario pendiente
    if st.session_state.pending_modal:
        show_toxicity_modal()

    col_main, col_right = st.columns([2.8, 1], gap="large")

    with col_main:
        # Video
        st.markdown(
            "<div class='video-thumb'><div class='play-btn'>▶</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='video-title'>AI Moderation Demo — Detección de Hate Speech en tiempo real</div>"
            "<div class='video-meta'>15k vistas · 2 horas atrás</div>",
            unsafe_allow_html=True,
        )
        row_ch, row_sub = st.columns([3, 1])
        with row_ch:
            st.markdown(
                "<div style='display:flex; align-items:center; gap:10px; margin:10px 0;'>"
                "<div style='width:36px; height:36px; border-radius:50%; background:#cc0000;"
                "display:flex; align-items:center; justify-content:center; color:#fff;"
                "font-weight:700;'>S</div>"
                "<div><div class='channel-name'>SignalMod AI</div>"
                "<div class='video-meta'>1.2M suscriptores</div></div></div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Comentarios ────────────────────────────────────────────────────
        tox_cnt = sum(1 for c in st.session_state.comments if c["is_toxic"])
        st.markdown(
            f"<div class='sec-title'>{len(st.session_state.comments)} Comentarios "
            f"<span style='font-size:0.8rem; color:#cc0000;'>· {tox_cnt} detectados</span></div>",
            unsafe_allow_html=True,
        )

        # Input de nuevo comentario
        new_text = st.text_area(
            "Escribe un comentario...",
            height=80, label_visibility="collapsed",
            key="comment_input",
            placeholder="Escribe un comentario...",
        )

        # Análisis en tiempo real (solo cuando hay texto)
        analysis = None
        if new_text.strip():
            svc      = get_service(st.session_state.selected_model)
            analysis = svc.predict(new_text)
            pct      = int(analysis["probability"] * 100)
            color    = "#cc0000" if pct >= 70 else "#f5a623" if pct >= 40 else "#00c853"
            verdict  = "TÓXICO" if analysis["is_toxic"] else "SEGURO"
            v_color  = "#cc0000" if analysis["is_toxic"] else "#00c853"
            st.markdown(
                f"<div class='tox-row'>"
                f"<span>🔍 Analizando...</span>"
                f"<span style='background:{v_color}; color:#fff; border-radius:10px;"
                f"padding:1px 9px; font-size:0.72rem; font-weight:700;'>{verdict}</span>"
                f"<span style='color:{color}; font-weight:600;'>Toxicidad: {pct}%</span>"
                f"<div class='tox-bar-bg'>"
                f"<div class='tox-bar-fill' style='width:{pct}%; background:{color};'></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        col_c, col_p = st.columns([1, 1])
        with col_c:
            if st.button("Cancelar", use_container_width=True):
                st.rerun()
        with col_p:
            post = st.button("Comentar", type="primary", use_container_width=True)

        # Procesar envío
        if post and new_text.strip():
            if analysis is None:
                svc      = get_service(st.session_state.selected_model)
                analysis = svc.predict(new_text)

            comment_obj = {
                "user"       : "usuario",
                "initial"    : "U",
                "text"       : new_text.strip(),
                "time"       : "ahora",
                "is_toxic"   : analysis["is_toxic"],
                "probability": analysis["probability"],
                "labels"     : analysis["labels"],
            }

            if analysis["is_toxic"]:
                # Guardar en pendiente y mostrar modal en el próximo render
                st.session_state.pending_modal = comment_obj
                st.rerun()
            else:
                # Publicar directamente
                st.session_state.comments.append(comment_obj)
                st.session_state.hub_history.insert(0, {
                    "Usuario"   : "@usuario",
                    "Comentario": f'"{new_text.strip()[:45]}{"..." if len(new_text)>45 else ""}"',
                    "Score"     : round(analysis["probability"], 2),
                    "Acción"    : "✅ Aprobado",
                })
                st.rerun()

        # ── Lista de comentarios ───────────────────────────────────────────
        for c in reversed(st.session_state.comments):
            is_tox   = c["is_toxic"]
            pct      = int(c["probability"] * 100)
            av_class = "c-avatar" if is_tox else "c-avatar safe"
            badge    = (
                "<span class='badge badge-toxic'>TÓXICO</span>" if is_tox
                else "<span class='badge badge-safe'>SEGURO</span>"
            )
            text_class = "c-text toxic" if is_tox else "c-text"
            flagged    = "<div class='c-flagged'>🚩 Flagged for review</div>" if is_tox else ""

            # html.escape() protege contra caracteres que rompen el HTML
            safe_text = html.escape(c["text"])
            safe_user = html.escape(c["user"])
            initial   = html.escape(c.get("initial", c["user"][0].upper()))

            st.markdown(
                f"<div class='comment-wrap'>"
                f"  <div class='{av_class}'>{initial}</div>"
                f"  <div class='c-body'>"
                f"    <div class='c-header'>"
                f"      <span class='c-user'>@{safe_user}</span>"
                f"      <span class='c-time'>{c['time']}</span>"
                f"      {badge}"
                f"    </div>"
                f"    <div class='{text_class}'>{safe_text}</div>"
                f"    {flagged}"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Columna derecha ────────────────────────────────────────────────────
    with col_right:
        st.markdown("**Sugeridos**")
        suggested = [
            ("🤖", "Understanding Transformer Models...", "Neural Systems", "89k · 1 día"),
            ("🎓", "The Future of Content Moderation",   "Tech Ethics Pro", "1.4M · 2 sem"),
            ("📡", "Signal vs Noise: SignalMod Deep Dive","SignalMod AI",    "250k · 3 días"),
            ("💡", "Why AI Moderation is Harder Than...", "Ethics in Code",  "45k · 5 h"),
            ("🔬", "Hate Speech Detection 2024",          "AI Research Lab", "12k · 1 sem"),
        ]
        for emoji, title, ch, meta in suggested:
            st.markdown(
                f"<div class='sug-card'>"
                f"  <div class='sug-thumb'>{emoji}</div>"
                f"  <div>"
                f"    <div class='sug-title'>{html.escape(title)}</div>"
                f"    <div class='sug-ch'>{html.escape(ch)}</div>"
                f"    <div class='sug-meta'>{html.escape(meta)}</div>"
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODERATOR HUB
# ══════════════════════════════════════════════════════════════════════════════
def render_hub():
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("Instala plotly: pip install plotly")
        return

    st.markdown("## 📊 Panel de Estadísticas")

    # ── Cards de configuración ──────────────────────────────────────────────
    model_short = st.session_state.selected_model.split("(")[0].strip()
    c1, c2, c3 = st.columns(3)
    for col, label, val in [
        (c1, "MODEL ARCHITECTURE",  model_short),
        (c2, "CONFIDENCE THRESHOLD", f"{st.session_state.threshold:.2f} Alpha"),
        (c3, "LANGUAGE COVERAGE",    "English"),
    ]:
        with col:
            st.markdown(
                f"<div class='hub-card'>"
                f"<div class='hub-kpi-label'>{label}</div>"
                f"<div style='font-weight:600; font-size:0.95rem; color:#0f0f0f;'>"
                f"{html.escape(str(val))}</div></div>",
                unsafe_allow_html=True,
            )

    st.write("")

    # ── KPIs ───────────────────────────────────────────────────────────────
    total    = len(st.session_state.comments) + 100
    tox_cnt  = sum(1 for c in st.session_state.comments if c["is_toxic"]) + 5
    tox_rate = tox_cnt / total * 100
    m1, m2, m3 = st.columns(3)
    m1.metric("💬 Total comentarios", f"{total:,}", "+12%")
    m2.metric("☠️ Tasa de toxicidad",  f"{tox_rate:.1f}%",
              f"+0.8%", delta_color="inverse")
    m3.metric("🎯 F1 Score",           "0.7579", "Stable")

    st.divider()

    # ── Gráficos ───────────────────────────────────────────────────────────
    gcol, pcol = st.columns([2.2, 1])

    with gcol:
        days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
        vals = [random.randint(30, 80) for _ in days]
        vals[3] = max(vals) + 25
        colors = ["#cc0000" if i == 3 else "#b3c6ff" for i in range(7)]
        fig = go.Figure(go.Bar(x=days, y=vals, marker_color=colors, width=0.55))
        fig.update_layout(
            title="Tendencias de Toxicidad (7D)",
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            margin=dict(l=20, r=20, t=40, b=20), height=260,
            font=dict(size=11, color="#0f0f0f"),
        )
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with pcol:
        fig2 = go.Figure(go.Pie(
            labels=["Hate Speech","Insulto","Agresividad"],
            values=[45, 35, 20],
            hole=0.58,
            marker_colors=["#cc0000","#0f0f0f","#909090"],
            textfont_size=11,
        ))
        fig2.update_layout(
            title="Categorías",
            paper_bgcolor="#ffffff",
            margin=dict(l=10, r=10, t=40, b=10), height=260,
            legend=dict(font=dict(size=10), orientation="v"),
            font=dict(size=11, color="#0f0f0f"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Historial ──────────────────────────────────────────────────────────
    st.markdown("### Historial Reciente")
    df = pd.DataFrame(st.session_state.hub_history)
    if not df.empty:
        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.2f"
                )
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
def render_settings():
    st.markdown("## ⚙️ Ajustes")

    # ── Selección de modelo ─────────────────────────────────────────────────
    st.markdown("### 🤖 Modelo de detección",)
    st.caption(
        "Los modelos HuggingFace se descargan la primera vez (~300–600 MB). "
        "Requieren: `pip install transformers torch sentencepiece`"
    )
    st.write("")

    # Usamos st.radio para la selección — sin bugs de HTML
    model_names = list(AVAILABLE_MODELS.keys())
    current_idx = model_names.index(st.session_state.selected_model) \
                  if st.session_state.selected_model in model_names else 0

    chosen = st.radio(
        "Seleccionar modelo",
        model_names,
        index=current_idx,
        label_visibility="collapsed",
    )

    if chosen != st.session_state.selected_model:
        st.session_state.selected_model = chosen
        st.rerun()

    # Ficha del modelo seleccionado
    info = AVAILABLE_MODELS[st.session_state.selected_model]
    st.markdown(
        f"<div class='model-card active'>"
        f"<div class='model-card-name'>{info['icon']}  {html.escape(st.session_state.selected_model)}</div>"
        f"<div class='model-card-desc'>{html.escape(info['description'])}</div>"
        f"<div style='margin-top:8px;'>"
        f"<span class='model-pill'>⚡ {html.escape(info['speed'])}</span>"
        f"<span class='model-pill'>🎯 {html.escape(info['accuracy'])}</span>"
        f"<span class='model-pill'>📦 {html.escape(info['requires'])}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Info sobre modelo fine-tuneado
    if st.session_state.selected_model == "Modelo fine-tuneado (local)":
        path = PROJECT_ROOT / "models" / "finetuned_hf"
        if path.exists():
            st.success(f"✅ Modelo encontrado en `{path}`")
        else:
            st.warning(
                f"⚠️ No se encontró el modelo en `{path}`. "
                f"Ejecuta el **notebook 08** para generar el modelo fine-tuneado."
            )

    st.divider()

    # ── Umbral de confianza ─────────────────────────────────────────────────
    st.markdown("### 🎚️ Umbral de confianza")
    st.caption("Probabilidad mínima para marcar un comentario como tóxico.")

    new_thr = st.slider(
        "Umbral",
        min_value=0.3, max_value=0.9, step=0.05,
        value=st.session_state.threshold,
        label_visibility="collapsed",
        format="%.2f",
    )
    if new_thr != st.session_state.threshold:
        st.session_state.threshold = new_thr
        st.info(f"Umbral actualizado: **{new_thr:.2f}**")

    ta, tb = st.columns(2)
    ta.info(f"⬇️ **{new_thr:.2f}** bajo → más FP (más censura)", icon="⚠️")
    tb.info(f"⬆️ **{new_thr:.2f}** alto → más FN (más escapes)", icon="⚠️")

    st.divider()

    # ── Test rápido ─────────────────────────────────────────────────────────
    st.markdown("### 🧪 Probar modelo")
    test_txt = st.text_input(
        "Texto a analizar",
        placeholder="Ej: This is absolutely stupid and racist...",
        label_visibility="collapsed",
    )
    if st.button("Analizar", type="primary") and test_txt.strip():
        with st.spinner("Analizando..."):
            svc = get_service(st.session_state.selected_model)
            res = svc.predict(test_txt)

        pct     = int(res["probability"] * 100)
        verdict = "🔴 TÓXICO" if res["is_toxic"] else "🟢 SEGURO"
        st.markdown(f"**{verdict}** — {pct}% de toxicidad")
        st.progress(res["probability"])
        if res["labels"]:
            st.markdown(f"**Categorías:** {', '.join(res['labels'])}")
        if "error" in res:
            st.error(f"Error: {res['error']}")
        st.caption(f"Modelo: {res['model_used']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_sidebar()

    page = st.session_state.page
    if page == "Home":
        render_home()
    elif page == "Moderator Hub":
        render_hub()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()