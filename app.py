"""Solar MultiModel — três modelos fotovoltaicos em uma única interface."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, time

import numpy as np
import pandas as pd
import streamlit as st

from config.pv_database import MODULE_DB, get_module
from config.settings import PROFILES
from models.pv_module import ModuleSTC, SDMParams
from simulation.multimodel import (
    DEFAULT_EXPORT_COLUMNS,
    MODEL_LABELS,
    MODEL_NOCT,
    MODEL_ORDER,
    MODEL_SDM,
    MODEL_SHORT_LABELS,
    MODEL_SIMPLE,
    available_export_columns,
    build_export_dataframe,
    build_synthetic_profile,
    candidate_window_starts,
    compute_model_kpis,
    detect_input_columns,
    normalize_filename,
    prepare_uploaded_profile,
    read_input_csv,
    run_all_models,
)
from simulation.solver import extract_sdm_params
from visualization.multimodel_plots import (
    plot_comparison_efficiency,
    plot_comparison_energy,
    plot_comparison_power,
    plot_cumulative_energy,
    plot_difference_to_reference,
    plot_efficiency,
    plot_input_profile,
    plot_iv_pv_at_peak,
    plot_model_power,
    plot_temperatures,
)


st.set_page_config(
    page_title="Solar MultiModel",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
  :root {
    --solar-navy: #25394B;
    --solar-navy-dark: #1C2C3A;
    --solar-blue: #00699A;
    --solar-blue-light: #E9F5FB;
    --solar-border: #D8DEE5;
    --solar-text: #17222D;
    --solar-muted: #718096;
  }

  .stApp { background: #FFFFFF; color: var(--solar-text); }
  .block-container {
    padding-top: 4.75rem; padding-bottom: 3.5rem;
    padding-left:1.25rem; padding-right:1.6rem; max-width: 1780px;
  }
  h1, h2, h3 { letter-spacing: -0.025em; color: var(--solar-text); }

  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--solar-navy-dark) 0%, var(--solar-navy) 100%);
    border-right: 1px solid #324B60;
  }
  [data-testid="stSidebar"] > div { padding-top: 1.0rem; }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #EAF1F6 !important; }
  [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }

  .solar-brand { text-align: center; padding: 1.2rem .3rem 1.45rem; }
  .solar-mark {
    width: 70px; height: 70px; margin: 0 auto .8rem;
    display: grid; place-items: center; border-radius: 50%;
    color: #FFFFFF; font-size: 2.3rem;
    background: radial-gradient(circle at 42% 42%, #FFD76A 0 12%, #F5A623 42%, transparent 44%);
    border: 1px solid rgba(255,255,255,.28);
  }
  .solar-brand-name { color: white; font-weight: 850; letter-spacing: .23em; font-size: 1.02rem; }
  .solar-brand-sub { color: #91A9BB; font-weight: 700; letter-spacing: .12em; font-size: .64rem; margin-top: .35rem; }
  .sidebar-label { color:#91A9BB; font-size:.67rem; font-weight:800; letter-spacing:.13em; margin: .25rem 0 .55rem; }
  .sidebar-status {
    border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.055);
    border-radius:9px; padding:.72rem .8rem; margin-bottom:.55rem;
  }
  .sidebar-status small { display:block; color:#91A9BB !important; font-size:.65rem; letter-spacing:.1em; font-weight:800; }
  .sidebar-status b { display:block; color:#FFFFFF !important; font-size:.82rem; margin-top:.18rem; }

  .page-head {
    border: 1px solid var(--solar-border); border-radius: 10px; background:#FFFFFF;
    padding: .72rem .95rem; margin-bottom: .62rem;
  }
  .eyebrow { color: var(--solar-blue); font-size:.68rem; font-weight:850; letter-spacing:.14em; text-transform:uppercase; }
  .page-title { font-size:1.55rem; line-height:1.08; font-weight:900; color:#111A22; margin:.2rem 0 .12rem; }
  .page-subtitle { color:var(--solar-muted); font-size:.82rem; }
  .panel-title { color:var(--solar-blue); font-size:.66rem; font-weight:850; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.38rem; }
  .formula-box {
    background:#F7FAFC; border:1px solid #E1E7ED; border-left:3px solid var(--solar-blue);
    border-radius:8px; padding:.5rem .72rem; color:#3E4C59; font-size:.74rem; margin:.16rem 0 .52rem;
  }
  .status-row { display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.55rem; margin-bottom:.55rem; }
  .chip { display:inline-flex; align-items:center; border-radius:999px; padding:.28rem .56rem; font-size:.67rem; font-weight:800; }
  .chip-ok { background:#E8F7F1; color:#087A55; border:1px solid #BCE9D8; }
  .chip-warn { background:#FFF4DF; color:#9A6200; border:1px solid #F3D494; }
  .chip-info { background:#EAF5FB; color:#006390; border:1px solid #BBDCEC; }
  .datasheet-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.5rem; margin:.45rem 0 .2rem; }
  .datasheet-item { border:1px solid #E1E6EB; border-radius:8px; padding:.58rem .65rem; background:#FBFCFD; }
  .datasheet-item small { display:block; color:#7B8894; font-size:.61rem; letter-spacing:.08em; font-weight:800; text-transform:uppercase; }
  .datasheet-item b { display:block; color:#17222D; font-size:.86rem; margin-top:.12rem; }
  .model-status-card { border:1px solid #DDE3E9; border-radius:9px; padding:.52rem .65rem; background:#FFFFFF; min-height:64px; }
  .model-status-card b { color:#17222D; font-size:.8rem; }
  .model-status-card p { color:#718096; font-size:.7rem; margin:.22rem 0 0; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.35rem; }
  .compact-note {
    min-height:236px; display:flex; flex-direction:column; justify-content:center;
    border:1px dashed #C9D9E4; border-radius:8px; background:#F7FAFC;
    padding:1rem 1.1rem; color:#536575; font-size:.78rem; line-height:1.55;
  }
  .compact-note b { color:#203746; font-size:.85rem; margin-bottom:.25rem; }
  .model-page-title { color:#111A22; font-size:1.42rem; line-height:1.08; font-weight:900; margin:.2rem 0 .3rem; }
  .model-page-subtitle { color:#718096; font-size:.75rem; line-height:1.45; margin-bottom:.22rem; }
  .config-facts { display:grid; grid-template-columns:1fr 1fr; gap:.42rem; margin:.18rem 0 .12rem; }
  .config-fact { border:1px solid #E0E6EB; border-radius:7px; background:#FBFCFD; padding:.46rem .52rem; }
  .config-fact small { display:block; color:#788896; font-size:.57rem; font-weight:850; letter-spacing:.09em; text-transform:uppercase; }
  .config-fact b { display:block; color:#20313F; font-size:.72rem; margin-top:.12rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  .overview-lead { color:#3E4C59; font-size:.93rem; line-height:1.72; margin:.15rem 0 .8rem; }
  .overview-lead b { color:#17222D; }
  .overview-note {
    border:1px solid #CFE3EF; background:#F1F8FC; border-radius:9px;
    padding:.72rem .82rem; color:#31536A; font-size:.78rem; line-height:1.55;
  }
  [data-testid="stImage"] img { border-radius:9px; border:1px solid #DCE5EC; }
  .explain-tag {
    display:inline-flex; align-items:center; border-radius:999px; padding:.26rem .55rem;
    font-size:.62rem; font-weight:850; letter-spacing:.09em; text-transform:uppercase;
    margin-bottom:.35rem;
  }
  .tag-blue { color:#1766A3; background:#E9F3FC; border:1px solid #C6DFF3; }
  .tag-green { color:#087A55; background:#E8F7F1; border:1px solid #BCE9D8; }
  .tag-orange { color:#A55C00; background:#FFF3E2; border:1px solid #F2D3A5; }
  .model-explainer-title { color:#17222D; font-weight:900; font-size:1.02rem; margin:.1rem 0 .4rem; }
  .model-explainer-text { color:#5F6F7E; font-size:.78rem; line-height:1.55; min-height:74px; }
  .needs-line { color:#748391; font-size:.68rem; margin-top:.55rem; margin-bottom:.55rem; }
  .needs-line b { color:#3A4C5C; }

  .flow-shell {
    border:1px solid #DCE3E9; border-radius:10px; background:#FAFCFD;
    padding:1rem; text-align:center;
  }
  .flow-node {
    border:1px solid #C7D9E5; border-radius:9px; background:#FFFFFF;
    padding:.68rem .75rem; color:#17222D;
  }
  .flow-node small { display:block; color:#08729E; font-size:.61rem; font-weight:850; letter-spacing:.11em; margin-bottom:.2rem; }
  .flow-node b { display:block; font-size:.8rem; }
  .flow-arrow { color:#3B7A9D; font-size:1.35rem; line-height:1; padding:.3rem 0; font-weight:900; }
  .flow-parallel { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem; }
  .flow-model { border-radius:9px; padding:.72rem .58rem; background:#FFFFFF; }
  .flow-model b { display:block; color:#17222D; font-size:.76rem; }
  .flow-model span { display:block; color:#70808E; font-size:.65rem; margin-top:.18rem; }
  .flow-blue { border:1px solid #BFD9F0; border-top:3px solid #2F80ED; }
  .flow-green { border:1px solid #BDE5D7; border-top:3px solid #16A085; }
  .flow-orange { border:1px solid #F2D2A9; border-top:3px solid #F2994A; }
  .flow-outputs { display:grid; grid-template-columns:1fr 1fr; gap:.65rem; }

  .reliability-list { margin:.15rem 0 0; padding-left:1.05rem; color:#536575; font-size:.8rem; line-height:1.65; }
  .reliability-list li { margin-bottom:.32rem; }
  .reliability-list b { color:#243746; }
  .availability-grid { display:grid; gap:.55rem; }
  .availability-row {
    display:grid; grid-template-columns:1.05fr .95fr; gap:.7rem; align-items:center;
    border:1px solid #DEE5EA; border-radius:8px; padding:.65rem .7rem; background:#FBFCFD;
  }
  .availability-row small { display:block; color:#7A8996; font-size:.6rem; font-weight:850; letter-spacing:.09em; }
  .availability-row b { color:#223542; font-size:.76rem; }
  .availability-result { text-align:right; font-size:.72rem; font-weight:850; }
  .result-ok { color:#087A55; }
  .result-warn { color:#A66A00; }
  .result-stop { color:#B33A3A; }
  div[data-testid="stLaTeX"] { font-size:.87rem; }

  div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--solar-border) !important; border-radius:10px !important;
    box-shadow:none !important; padding-bottom:.12rem;
  }
  div[data-testid="stVerticalBlock"] { gap:.72rem !important; }
  div[data-testid="stMetric"] {
    background:#FFFFFF; border:1px solid #DDE3E9; border-radius:9px;
    padding:.48rem .62rem; min-height:76px;
  }
  div[data-testid="stMetricLabel"] { color:#657484; font-size:.69rem; }
  div[data-testid="stMetricValue"] { color:#111820; font-size:1.12rem; font-weight:850; }
  div[data-testid="stMetricDelta"] { font-size:.62rem; }
  div[data-testid="stAlert"] { padding:.52rem .7rem; }
  div[data-testid="stPlotlyChart"] { margin:0 !important; }

  .stButton > button, .stDownloadButton > button {
    border-radius:8px; font-weight:800; min-height:2.65rem; box-shadow:none;
  }
  .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
    background:linear-gradient(135deg,#0878A8,#00577F); border-color:#00577F;
  }
  [data-testid="stSidebar"] .stButton { margin-bottom:.38rem; }
  [data-testid="stSidebar"] .stButton > button {
    width:100%; min-height:2.55rem; border-radius:9px;
    background:rgba(255,255,255,.065); border:1px solid rgba(255,255,255,.12);
    color:#EAF1F6; font-size:.79rem; font-weight:760; box-shadow:none;
    transition:background .16s ease,border-color .16s ease,transform .16s ease;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background:rgba(255,255,255,.11); border-color:rgba(255,255,255,.25);
    color:#FFFFFF; transform:translateY(-1px);
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#1D496D 0%,#123B5F 100%);
    border:1px solid #5D86A6; color:#FFFFFF;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.04);
  }
  [data-testid="stSidebar"] .stButton > button p { color:inherit !important; font-weight:inherit; }
  .stTabs [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #DDE3E9; }
  .stTabs [data-baseweb="tab"] { padding:.48rem .72rem; font-weight:750; }
  [data-baseweb="select"] > div, [data-baseweb="input"] > div { border-radius:8px; }
  [data-testid="stFileUploaderDropzone"] { border-radius:9px; background:#F8FAFC; }

  @media (max-width: 900px) {
    .datasheet-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .flow-parallel, .flow-outputs { grid-template-columns:1fr; }
    .availability-row { grid-template-columns:1fr; }
    .availability-result { text-align:left; }
    .block-container { padding-left:1rem; padding-right:1rem; }
  }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "solar_multimodel"},
}

NAV_OVERVIEW = "Visão geral"
NAV_INPUT = "Entrada"
NAV_MODELS = "Modelos"
NAV_COMPARISON = "Comparação"
NAV_EXPORT = "Exportação"
NAV_OPTIONS = (NAV_OVERVIEW, NAV_INPUT, NAV_MODELS, NAV_COMPARISON, NAV_EXPORT)
NAV_KEYS = {
    NAV_OVERVIEW: "overview",
    NAV_INPUT: "input",
    NAV_MODELS: "models",
    NAV_COMPARISON: "comparison",
    NAV_EXPORT: "export",
}


def init_state() -> None:
    defaults = {
        "profile": None,
        "module": None,
        "results_by_model": {},
        "model_statuses": {},
        "kpis_by_model": {},
        "run_config": {},
        "extraction_report": None,
        "current_page": NAV_OVERVIEW,
        "last_run_notice": None,
        "selected_result_model": MODEL_SDM,
        "comparison_reference_model": MODEL_SDM,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


@st.cache_data(show_spinner=False)
def cached_sdm_extraction(stc_dict: dict) -> tuple[dict, dict]:
    stc = ModuleSTC(**stc_dict)
    params, report = extract_sdm_params(stc)
    return params.to_dict(), asdict(report)


def page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="eyebrow">{eyebrow}</div>
          <div class="page-title">{title}</div>
          <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def status_chip(text: str, kind: str = "info") -> str:
    return f'<span class="chip chip-{kind}">{text}</span>'


def _window_minutes(profile: pd.DataFrame) -> int:
    step = float(profile.attrs.get("timestep_minutes", 1.0))
    return int(round(len(profile) * step))


def _window_label(profile: pd.DataFrame) -> str:
    minutes = _window_minutes(profile)
    if minutes == 1440:
        return "24 h"
    if minutes == 120:
        return "120 min"
    if minutes % 60 == 0:
        return f"{minutes // 60} h"
    return f"{minutes} min"


def _window_slug(profile: pd.DataFrame) -> str:
    minutes = _window_minutes(profile)
    return "24h" if minutes == 1440 else f"{minutes}min"


def model_datasheet(module) -> None:
    stc = module.stc
    st.markdown(
        f"""
        <div class="datasheet-grid">
          <div class="datasheet-item"><small>Potência STC</small><b>{stc.p_nom:.0f} W</b></div>
          <div class="datasheet-item"><small>Eficiência STC</small><b>{stc.efficiency_stc*100:.2f} %</b></div>
          <div class="datasheet-item"><small>Área</small><b>{stc.area:.3f} m²</b></div>
          <div class="datasheet-item"><small>NOCT</small><b>{stc.noct:.1f} °C</b></div>
          <div class="datasheet-item"><small>γ Pmax</small><b>{stc.gamma_pmax_pct:.3f} %/°C</b></div>
          <div class="datasheet-item"><small>Células elétricas</small><b>{stc.n_cells}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="solar-brand">
              <div class="solar-mark">☀</div>
              <div class="solar-brand-name">SOLAR</div>
              <div class="solar-brand-sub">MULTI-MODEL ENGINE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-label">NAVEGAÇÃO</div>', unsafe_allow_html=True)
        nav = st.session_state["current_page"]
        for option in NAV_OPTIONS:
            clicked = st.button(
                option,
                key=f"nav_{NAV_KEYS[option]}",
                type="primary" if option == nav else "secondary",
                width="stretch",
            )
            if clicked and option != nav:
                st.session_state["current_page"] = option
                st.rerun()

        st.divider()
        st.markdown('<div class="sidebar-label">ESTADO DA EXECUÇÃO</div>', unsafe_allow_html=True)
        module = st.session_state["module"]
        profile = st.session_state["profile"]
        results = st.session_state["results_by_model"]
        module_name = module.stc.model if module is not None else "Ainda não selecionado"
        if profile is not None:
            start = profile.index.min()
            end = profile.index.max()
            if start.date() == end.date():
                interval = f"{start:%H:%M}–{end:%H:%M}"
            else:
                interval = f"{start:%d/%m %H:%M}–{end:%d/%m %H:%M}"
            window = f"{interval} · {_window_label(profile)}"
        else:
            window = "Aguardando entrada"
        st.markdown(
            f"""
            <div class="sidebar-status"><small>MÓDULO</small><b>{module_name}</b></div>
            <div class="sidebar-status"><small>JANELA</small><b>{window}</b></div>
            <div class="sidebar-status"><small>MODELOS DISPONÍVEIS</small><b>{len(results)} de 3</b></div>
            """,
            unsafe_allow_html=True,
        )
        if results:
            chips = []
            for model_id in MODEL_ORDER:
                kind = "ok" if model_id in results else "warn"
                label = MODEL_SHORT_LABELS[model_id]
                chips.append(status_chip(("● " if model_id in results else "○ ") + label, kind))
            st.markdown('<div class="status-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)

        st.divider()
        st.caption("CSV: 120 min · perfil sintético: 120 min ou 24 h · passo de 1 minuto.")
        return st.session_state["current_page"]


def _default_index(options: list, value) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def render_overview_page() -> None:
    page_header(
        "Visão geral · Fundamentos e arquitetura",
        "DO SOL À POTÊNCIA ELÉTRICA",
        "Um motor fotovoltaico multimodelo para estimativa, comparação e continuidade operacional.",
    )

    intro, visual = st.columns([1.22, 0.98], gap="medium")
    with intro:
        with st.container(border=True, height="stretch"):
            panel_title("Conceito · O que é um modelo solar?")
            st.markdown(
                """
                <div class="overview-lead">
                  Um <b>modelo solar fotovoltaico</b> é uma representação matemática que transforma
                  condições ambientais e informações do painel em uma estimativa da potência elétrica disponível.
                  Ele conecta o que chega do ambiente — principalmente <b>irradiância</b> e
                  <b>temperatura</b> — ao comportamento do módulo selecionado no datasheet.
                </div>
                <div class="overview-note">
                  Nesta plataforma, todos os modelos recebem a mesma <b>janela temporal selecionada</b>,
                  o mesmo arranjo e o mesmo módulo. Assim, qualquer diferença observada vem da
                  formulação matemática, e não de entradas diferentes.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="status-row">'
                + status_chip("Entrada comum", "info")
                + status_chip("3 modelos em paralelo", "ok")
                + status_chip("Saída rastreável", "info")
                + "</div>",
                unsafe_allow_html=True,
            )

    with visual:
        with st.container(border=True, height="stretch"):
            panel_title("Conversão fotovoltaica · Irradiância → eletricidade")
            st.image(
                "assets/fluxo_fotovoltaico.jpg",
                caption="A irradiância incidente é convertida pelo módulo fotovoltaico em potência elétrica.",
                width="stretch",
            )

    st.write("")
    panel_title("Três modelos · Mesma entrada, três níveis de complexidade")
    model_1, model_2, model_3 = st.columns(3, gap="medium")

    with model_1:
        with st.container(border=True, height="stretch"):
            st.markdown('<span class="explain-tag tag-blue">Modelo 1 · Continuidade</span>', unsafe_allow_html=True)
            st.markdown('<div class="model-explainer-title">Irradiância</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="model-explainer-text">Escala diretamente a potência de pico do datasheet '
                "pela irradiância efetiva. É rápido, transparente e não depende de temperatura.</div>",
                unsafe_allow_html=True,
            )
            st.latex(r"P_1(t)=N\,P_{STC}\,\frac{G_{ef}(t)}{G_{STC}}")
            st.markdown(
                '<div class="needs-line"><b>Precisa:</b> timestamp + GHI + potência STC.</div>',
                unsafe_allow_html=True,
            )

    with model_2:
        with st.container(border=True, height="stretch"):
            st.markdown('<span class="explain-tag tag-green">Modelo 2 · Térmico</span>', unsafe_allow_html=True)
            st.markdown('<div class="model-explainer-title">NOCT + eficiência</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="model-explainer-text">Calcula a temperatura da célula com o NOCT e '
                "corrige a eficiência pelo coeficiente térmico de potência publicado no datasheet.</div>",
                unsafe_allow_html=True,
            )
            st.latex(r"T_c=T_{amb}+\frac{NOCT-20}{800}G_{ef}")
            st.latex(r"P_2=N\,\eta_{STC}[1+\gamma_P(T_c-25)]\,G_{ef}A")
            st.markdown(
                '<div class="needs-line"><b>Precisa:</b> GHI + Tamb + NOCT + área + γPmax.</div>',
                unsafe_allow_html=True,
            )

    with model_3:
        with st.container(border=True, height="stretch"):
            st.markdown('<span class="explain-tag tag-orange">Modelo 3 · Físico</span>', unsafe_allow_html=True)
            st.markdown('<div class="model-explainer-title">Single Diode Model</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="model-explainer-text">Representa eletricamente o módulo, translada os '
                "cinco parâmetros para cada condição e resolve numericamente o ponto de máxima potência.</div>",
                unsafe_allow_html=True,
            )
            st.latex(
                r"I=I_L-I_0\!\left[e^{\frac{V+IR_s}{a}}-1\right]-\frac{V+IR_s}{R_{sh}}"
            )
            st.latex(r"P_3(t)=N\,\max_V\{V\,I(V)\}")
            st.markdown(
                '<div class="needs-line"><b>Precisa:</b> GHI + Tamb + parâmetros elétricos do módulo.</div>',
                unsafe_allow_html=True,
            )

    st.write("")
    with st.container(border=True):
        panel_title("Fluxo de funcionamento · Execução paralela")
        st.markdown(
            """
            <div class="flow-shell">
              <div class="flow-node">
                <small>ENTRADA COMUM · 120 MINUTOS OU 24 HORAS</small>
                <b>timestamp · GHI · Tamb opcional · datasheet · arranjo 2S × 3P</b>
              </div>
              <div class="flow-arrow">↓</div>
              <div class="flow-parallel">
                <div class="flow-model flow-blue">
                  <b>Modelo 1 · Irradiância</b>
                  <span>estimativa de continuidade</span>
                </div>
                <div class="flow-model flow-green">
                  <b>Modelo 2 · NOCT</b>
                  <span>estimativa térmica</span>
                </div>
                <div class="flow-model flow-orange">
                  <b>Modelo 3 · SDM</b>
                  <span>estimativa físico-elétrica</span>
                </div>
              </div>
              <div class="flow-arrow">↓</div>
              <div class="flow-outputs">
                <div class="flow-node">
                  <small>CONSISTÊNCIA</small>
                  <b>Sobreposição, divergência e comparação dos resultados</b>
                </div>
                <div class="flow-node">
                  <small>CONTINUIDADE</small>
                  <b>Operação degradada com o Modelo 1 quando Tamb é perdida</b>
                </div>
              </div>
              <div class="flow-arrow">↓</div>
              <div class="flow-node">
                <small>SAÍDA</small>
                <b>Modelo selecionado + colunas escolhidas para exportação</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    reliability, availability = st.columns([1.05, .95], gap="medium")
    with reliability:
        with st.container(border=True):
            panel_title("Confiabilidade · Por que manter três modelos?")
            st.markdown(
                """
                <ul class="reliability-list">
                  <li><b>Degradação controlada:</b> se a temperatura deixa de chegar, o modelo simples continua entregando uma referência de potência.</li>
                  <li><b>Redundância analítica:</b> três formulações independentes ajudam a identificar resultados anômalos ou divergências inesperadas.</li>
                  <li><b>Rastreabilidade:</b> a plataforma informa quais modelos executaram; não inventa uma temperatura para esconder a falha da entrada.</li>
                  <li><b>Validação progressiva:</b> o modelo simples funciona como linha de base, o intermediário isola o efeito térmico e o SDM representa a física elétrica completa.</li>
                  <li><b>Continuidade do EMS:</b> o otimizador pode receber ao menos uma orientação de geração mesmo quando a medição de Tamb está indisponível.</li>
                </ul>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="overview-note">
                  <b>Importante:</b> os três modelos compartilham a irradiância. Portanto, a arquitetura
                  aumenta a disponibilidade diante da perda de temperatura e permite verificação cruzada,
                  mas não elimina uma falha da própria GHI. Isso exigiria redundância também na aquisição ou previsão meteorológica.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with availability:
        with st.container(border=True):
            panel_title("Disponibilidade · Resposta do motor")
            st.markdown(
                """
                <div class="availability-grid">
                  <div class="availability-row">
                    <div><small>ENTRADA COMPLETA</small><b>GHI + Tamb válidas</b></div>
                    <div class="availability-result result-ok">3 modelos executados</div>
                  </div>
                  <div class="availability-row">
                    <div><small>MODO DEGRADADO</small><b>GHI válida · Tamb ausente</b></div>
                    <div class="availability-result result-warn">Modelo 1 continua</div>
                  </div>
                  <div class="availability-row">
                    <div><small>ENTRADA CRÍTICA</small><b>GHI ausente ou janela irregular</b></div>
                    <div class="availability-result result-stop">Execução bloqueada</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                """
                <div class="formula-box">
                  <b>Confiabilidade aqui significa disponibilidade + diagnóstico.</b><br>
                  A comparação não é o único objetivo: ela também permite detectar quando uma
                  estimativa se afasta das outras e manter uma saída útil em condição degradada.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    if st.button(
        "→ IR PARA ENTRADA E CONFIGURAR UMA JANELA",
        type="primary",
        width="stretch",
        key="overview_to_input",
    ):
        st.session_state["current_page"] = NAV_INPUT
        st.rerun()


def render_input_page() -> None:
    page_header(
        "Entrada · Configuração comum",
        "ENTRADA DOS MODELOS",
        "Selecione o módulo, carregue um CSV ou gere uma janela sintética e execute os três modelos.",
    )

    run_requested = st.button(
        "▶ RODAR MODELOS",
        type="primary",
        width="stretch",
        key="run_models_button",
    )
    run_feedback = st.empty()
    if run_requested:
        st.session_state["last_run_notice"] = None
    else:
        previous_notice = st.session_state.get("last_run_notice")
        if previous_notice:
            getattr(run_feedback, previous_notice["kind"])(previous_notice["message"])

    def show_run_notice(kind: str, message: str) -> None:
        st.session_state["last_run_notice"] = {"kind": kind, "message": message}
        getattr(run_feedback, kind)(message)

    st.caption(
        "Configure o módulo e a fonte de dados abaixo. O botão executa a janela atualmente preparada."
    )

    left, right = st.columns([0.82, 1.48], gap="medium")
    preview_profile = None
    source_description = "—"

    with left:
        with st.container(border=True):
            panel_title("Sistema fotovoltaico · Datasheet")
            module_keys = list(MODULE_DB.keys())
            default_module = "CS7L-580MS"
            module_key = st.selectbox(
                "Módulo fotovoltaico",
                module_keys,
                index=_default_index(module_keys, default_module),
                help="Os três modelos recebem os parâmetros do mesmo datasheet.",
            )
            module_preview = get_module(module_key)

            a1, a2 = st.columns(2)
            with a1:
                n_series = st.number_input(
                    "Módulos em série", min_value=1, max_value=30, value=2, step=1
                )
            with a2:
                n_parallel = st.number_input(
                    "Strings em paralelo", min_value=1, max_value=30, value=3, step=1
                )
            losses_pct = st.slider(
                "Perdas ópticas / sujeira [%]", 0.0, 20.0, 0.0, 0.5
            )
            model_datasheet(module_preview)
            installed_kwp = module_preview.stc.p_nom * int(n_series) * int(n_parallel) / 1000.0
            st.markdown(
                '<div class="status-row">'
                + status_chip(f"{int(n_series)}S × {int(n_parallel)}P", "info")
                + status_chip(f"{int(n_series)*int(n_parallel)} módulos", "info")
                + status_chip(f"{installed_kwp:.3f} kWp", "ok")
                + "</div>",
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            panel_title("Hipótese comum")
            st.markdown(
                """
                <div class="formula-box">
                  A coluna selecionada como GHI é usada pelos três modelos como irradiância incidente no plano do módulo.
                  O arranjo, as perdas e o datasheet são idênticos para tornar a comparação consistente.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            panel_title("Fonte dos dados · Janela temporal")
            source = st.radio(
                "Fonte",
                options=("Carregar CSV", "Perfil sintético"),
                horizontal=True,
                label_visibility="collapsed",
            )

            if source == "Carregar CSV":
                uploaded = st.file_uploader(
                    "CSV minuto a minuto",
                    type=["csv"],
                    help="É necessário haver 120 timestamps consecutivos com passo de 1 minuto.",
                )
                if uploaded is None:
                    st.info("Carregue um CSV para detectar automaticamente timestamp, irradiância e temperatura.")
                    try:
                        with open("Dados_exemplo/PREVISAO_SOLAR_120min_12.csv", "rb") as example_file:
                            example_bytes = example_file.read()
                        st.download_button(
                            "⬇️ Baixar CSV de exemplo",
                            data=example_bytes,
                            file_name="PREVISAO_SOLAR_120min_EXEMPLO.csv",
                            mime="text/csv",
                            width="stretch",
                        )
                    except OSError:
                        pass
                else:
                    try:
                        raw = read_input_csv(uploaded)
                        detected = detect_input_columns(raw)
                        columns = list(raw.columns)
                        ts_default = detected.get("timestamp_default") or columns[0]
                        irr_candidates = detected.get("irradiance_candidates") or [
                            col for col in columns if col != ts_default
                        ]
                        irr_default = detected.get("irradiance_default") or irr_candidates[0]
                        detected_temp = detected.get("temperature_default")

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            timestamp_col = st.selectbox(
                                "Timestamp",
                                columns,
                                index=_default_index(columns, ts_default),
                            )
                        with c2:
                            irradiance_col = st.selectbox(
                                "Irradiância",
                                columns,
                                index=_default_index(columns, irr_default),
                            )
                        temp_options = ["Sem temperatura"] + [
                            col for col in columns if col not in {timestamp_col, irradiance_col}
                        ]
                        with c3:
                            temperature_choice = st.selectbox(
                                "Temperatura ambiente",
                                temp_options,
                                index=_default_index(temp_options, detected_temp),
                            )
                        temperature_col = None if temperature_choice == "Sem temperatura" else temperature_choice

                        starts = candidate_window_starts(raw, timestamp_col)
                        if not starts:
                            raise ValueError(
                                "O arquivo não contém dados suficientes para uma janela de 120 minutos."
                            )
                        start_ts = st.selectbox(
                            "Início da janela",
                            starts,
                            index=0,
                            format_func=lambda ts: ts.strftime("%Y-%m-%d %H:%M:%S"),
                        )
                        preview_profile = prepare_uploaded_profile(
                            raw,
                            timestamp_col=timestamp_col,
                            irradiance_col=irradiance_col,
                            temperature_col=temperature_col,
                            start=start_ts,
                        )
                        source_description = f"CSV · {uploaded.name}"
                    except Exception as exc:
                        st.error(f"Não foi possível preparar o CSV: {exc}")
                        preview_profile = None

            else:
                s1, s2, s3 = st.columns(3)
                with s1:
                    synthetic_date = st.date_input("Data", value=date.today())
                with s2:
                    synthetic_time = st.time_input("Hora inicial", value=time(12, 0))
                with s3:
                    duration_minutes = st.selectbox(
                        "Janela sintética",
                        options=(120, 1440),
                        format_func=lambda value: (
                            "2 horas · 120 min" if value == 120 else "1 dia · 24 h"
                        ),
                    )

                profile_options = [
                    "Irradiância perfeita",
                    "Día soleado",
                    "Día nublado",
                    "Día lluvioso",
                ]
                profile_display = {
                    "Irradiância perfeita": "Irradiância perfeita · curva suave",
                    "Día soleado": "Dia ensolarado",
                    "Día nublado": "Dia nublado",
                    "Día lluvioso": "Dia chuvoso",
                }
                s4, s5 = st.columns(2)
                with s4:
                    irradiance_profile = st.selectbox(
                        "Condição solar",
                        profile_options,
                        format_func=lambda x: profile_display[x],
                        help=(
                            "Irradiância perfeita usa somente a curva solar suave, "
                            "sem ruído nem quedas abruptas."
                        ),
                    )
                with s5:
                    season = st.selectbox("Estação", list(PROFILES["seasons"].keys()), index=0)

                defaults = {
                    "Irradiância perfeita": PROFILES["g_peak_clear"],
                    "Día soleado": PROFILES["g_peak_clear"],
                    "Día nublado": PROFILES["g_peak_cloudy"],
                    "Día lluvioso": PROFILES["g_peak_rainy"],
                }
                season_cfg = PROFILES["seasons"][season]
                p1, p2, p3 = st.columns(3)
                with p1:
                    g_peak = st.number_input(
                        "GHI pico [W/m²]", min_value=0.0, max_value=1400.0,
                        value=float(defaults[irradiance_profile]), step=10.0,
                    )
                with p2:
                    t_min = st.number_input(
                        "Temperatura mínima [°C]", value=float(season_cfg["t_min"]), step=0.5
                    )
                with p3:
                    t_max = st.number_input(
                        "Temperatura máxima [°C]", value=float(season_cfg["t_max"]), step=0.5
                    )
                if t_max < t_min:
                    st.error("A temperatura máxima deve ser maior ou igual à mínima.")
                else:
                    synthetic_start = pd.Timestamp.combine(synthetic_date, synthetic_time)
                    preview_profile = build_synthetic_profile(
                        start=synthetic_start,
                        irradiance_profile=irradiance_profile,
                        season=season,
                        duration_minutes=int(duration_minutes),
                        g_peak=float(g_peak),
                        t_min=float(t_min),
                        t_max=float(t_max),
                    )
                    source_description = (
                        f"Sintético · {profile_display[irradiance_profile]} · "
                        f"{_window_label(preview_profile)}"
                    )

            if preview_profile is not None:
                temp_ok = bool(preview_profile["Tamb"].notna().all())
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Linhas", f"{len(preview_profile)}")
                q2.metric("Passo", "1 min")
                q3.metric("GHI máxima", f"{preview_profile['G'].max():.1f} W/m²")
                q4.metric("Temperatura", "Disponível" if temp_ok else "Ausente")
                st.plotly_chart(
                    plot_input_profile(preview_profile),
                    width="stretch",
                    config=CHART_CONFIG,
                    key="input_profile_chart",
                )
                if temp_ok:
                    st.success("Entrada completa: os três modelos poderão ser executados.")
                else:
                    st.warning(
                        "Tamb não está completa. O modelo de irradiância continuará operando; "
                        "o modelo NOCT e o SDM serão marcados como indisponíveis."
                    )
    if run_requested and preview_profile is None:
        show_run_notice("warning", "Prepare uma janela válida antes de executar os modelos.")
    elif run_requested:
        try:
            module = get_module(module_key)
            extraction_report = None
            has_temperature = bool(preview_profile["Tamb"].notna().all())
            if has_temperature:
                with st.spinner("Extraindo os parâmetros SDM do datasheet…"):
                    sdm_dict, extraction_report = cached_sdm_extraction(module.stc.to_dict())
                module.sdm = SDMParams(**sdm_dict)
                if not extraction_report.get("success", False):
                    raise RuntimeError(
                        "A extração dos parâmetros SDM não convergiu para o módulo selecionado."
                    )

            progress = st.progress(0, text="Executando os modelos fotovoltaicos…")

            def update_progress(value: float) -> None:
                progress.progress(min(100, max(0, int(round(value * 100)))), text="Resolvendo o SDM minuto a minuto…")

            results, statuses = run_all_models(
                module,
                preview_profile,
                n_series=int(n_series),
                n_parallel=int(n_parallel),
                soiling_losses=float(losses_pct) / 100.0,
                noct=module.stc.noct,
                progress_callback=update_progress,
            )
            progress.empty()
            kpis = {
                model_id: compute_model_kpis(result, module)
                for model_id, result in results.items()
            }

            st.session_state["profile"] = preview_profile
            st.session_state["module"] = module
            st.session_state["results_by_model"] = results
            st.session_state["model_statuses"] = statuses
            st.session_state["kpis_by_model"] = kpis
            st.session_state["extraction_report"] = extraction_report
            st.session_state["run_config"] = {
                "source": source_description,
                "module_key": module_key,
                "n_series": int(n_series),
                "n_parallel": int(n_parallel),
                "soiling_losses_pct": float(losses_pct),
                "window_minutes": _window_minutes(preview_profile),
            }
            if len(results) == 3:
                show_run_notice(
                    "success",
                    "Execução concluída: os três modelos estão disponíveis nas abas de resultados.",
                )
            elif len(results) == 2:
                show_run_notice(
                    "warning",
                    "Execução concluída em modo degradado: 2 de 3 modelos estão disponíveis.",
                )
            else:
                show_run_notice(
                    "warning",
                    "Execução concluída em modo degradado: o modelo simples gerou a janela completa.",
                )
        except Exception as exc:
            show_run_notice("error", f"Não foi possível executar os modelos: {exc}")


def _empty_results(message: str) -> bool:
    if st.session_state["profile"] is None or not st.session_state["results_by_model"]:
        st.info(message)
        return True
    return False


def _metric_row(model_id: str, result: pd.DataFrame, kpi: dict) -> None:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Energia na janela", f"{kpi['energy_kWh']:.4f} kWh")
    m2.metric(
        "Potência máxima",
        f"{kpi['p_max_W']:.1f} W",
        f"pico às {kpi['t_peak']:%H:%M}",
        delta_color="off",
    )
    m3.metric("Potência média", f"{kpi['p_mean_W']:.1f} W")
    m4.metric("Eficiência energética", f"{kpi['eta_energy']*100:.2f} %")
    tc_label = "Não utilizada" if np.isnan(kpi["tc_max_C"]) else f"{kpi['tc_max_C']:.2f} °C"
    m5.metric("Temperatura de célula máxima", tc_label)


def _formula_for(model_id: str) -> None:
    if model_id == MODEL_SIMPLE:
        st.markdown(
            """
            <div class="formula-box"><b>Modelo 1:</b> P = P<sub>STC</sub> · G<sub>ef</sub>/1000.
            Usa somente irradiância, potência nominal e quantidade de módulos. A eficiência permanece constante.</div>
            """,
            unsafe_allow_html=True,
        )
    elif model_id == MODEL_NOCT:
        st.markdown(
            """
            <div class="formula-box"><b>Modelo 2:</b> Tc = Tamb + (NOCT − 20)·G<sub>ef</sub>/800;
            η(Tc) = η<sub>STC</sub>[1 + γ(Tc − 25)]; P = η(Tc)·G<sub>ef</sub>·A.</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="formula-box"><b>Modelo 3:</b> resolve a equação implícita do circuito equivalente de um diodo,
            translada os cinco parâmetros para (G,Tc) e encontra numericamente o MPP em cada minuto.</div>
            """,
            unsafe_allow_html=True,
        )


def render_models_page() -> None:
    if _empty_results("Execute uma janela na seção Entrada para visualizar os modelos."):
        return

    results_by_model = st.session_state["results_by_model"]
    kpis_by_model = st.session_state["kpis_by_model"]
    module = st.session_state["module"]
    profile = st.session_state["profile"]
    available_models = [model_id for model_id in MODEL_ORDER if model_id in results_by_model]
    if st.session_state.get("selected_result_model") not in available_models:
        st.session_state["selected_result_model"] = available_models[0]

    controls, main_chart = st.columns([0.62, 1.78], gap="small")
    with controls:
        with st.container(border=True):
            panel_title("Modelos · Resultados individuais")
            st.markdown(
                """
                <div class="model-page-title">MODELOS FOTOVOLTAICOS</div>
                <div class="model-page-subtitle">
                  Potência, energia e comportamento físico da mesma janela meteorológica.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True, height="stretch"):
            panel_title("Configurações · Modelo analisado")
            selected_model = st.selectbox(
                "Modelo analisado",
                options=available_models,
                format_func=lambda model_id: MODEL_LABELS[model_id],
                key="selected_result_model",
                help="Escolha qual dos modelos disponíveis alimentará todos os painéis desta página.",
            )
            st.markdown(
                f"""
                <div class="config-facts">
                  <div class="config-fact"><small>Módulo</small><b>{module.stc.model}</b></div>
                  <div class="config-fact"><small>Janela</small><b>{_window_label(profile)} · passo 1 min</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chips = []
            for model_id in MODEL_ORDER:
                available = model_id in results_by_model
                chips.append(
                    status_chip(
                        ("● " if available else "○ ") + MODEL_SHORT_LABELS[model_id],
                        "ok" if available else "warn",
                    )
                )
            st.markdown(
                '<div class="status-row">' + "".join(chips) + "</div>",
                unsafe_allow_html=True,
            )
            _formula_for(selected_model)

    result = results_by_model[selected_model]
    kpi = kpis_by_model[selected_model]
    with main_chart:
        with st.container(border=True, height="stretch"):
            panel_title("Síntese · Indicadores do modelo selecionado")
            _metric_row(selected_model, result, kpi)
            panel_title(f"Curva principal · Potência gerada · {MODEL_SHORT_LABELS[selected_model]}")
            st.plotly_chart(
                plot_model_power(result, selected_model, height=250),
                width="stretch",
                config=CHART_CONFIG,
                key=f"power_main_{selected_model}",
            )

    energy_col, thermal_col, efficiency_col = st.columns(3, gap="small")
    with energy_col:
        with st.container(border=True, height="stretch"):
            panel_title("Energia · Acumulada na janela")
            st.plotly_chart(
                plot_cumulative_energy(result, selected_model),
                width="stretch",
                config=CHART_CONFIG,
                key=f"energy_{selected_model}",
            )
    with thermal_col:
        with st.container(border=True, height="stretch"):
            panel_title("Comportamento térmico")
            if selected_model == MODEL_SIMPLE:
                st.markdown(
                    """
                    <div class="compact-note">
                      <b>Operação independente de temperatura</b>
                      Este modelo não calcula temperatura de célula. Por isso, continua
                      entregando potência quando a coluna Tamb fica indisponível.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.plotly_chart(
                    plot_temperatures(result),
                    width="stretch",
                    config=CHART_CONFIG,
                    key=f"temperature_{selected_model}",
                )
    with efficiency_col:
        with st.container(border=True, height="stretch"):
            panel_title("Eficiência · Evolução temporal")
            st.plotly_chart(
                plot_efficiency(result, selected_model),
                width="stretch",
                config=CHART_CONFIG,
                key=f"efficiency_{selected_model}",
            )

    if selected_model == MODEL_SDM:
        peak_ts = result["P_array"].idxmax()
        peak = result.loc[peak_ts]
        with st.expander(
            f"Diagnóstico elétrico avançado do SDM · pico às {peak_ts:%H:%M}",
            expanded=False,
        ):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Vmp do arranjo", f"{peak['Vmp_array']:.2f} V")
            e2.metric("Imp do arranjo", f"{peak['Imp_array']:.3f} A")
            e3.metric("Voc do arranjo", f"{peak['Voc_array']:.2f} V")
            e4.metric("Fator de forma", f"{peak['FF']:.4f}")
            if peak["G_eff"] > 0:
                panel_title("Curvas I-V e P-V no ponto de maior potência")
                st.plotly_chart(
                    plot_iv_pv_at_peak(module, result),
                    width="stretch",
                    config=CHART_CONFIG,
                    key="sdm_iv_pv_peak",
                )

    with st.expander(f"Ver tabela completa de resultados ({len(result)} linhas)"):
        preferred = [
            "G", "G_eff", "Tamb", "Tc", "P_module", "P_array", "eta",
            "Vmp", "Imp", "Voc", "Isc", "FF",
        ]
        cols = [col for col in preferred if col in result.columns]
        display = result[cols].copy()
        display.index = display.index.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display, width="stretch", height=310)


def _comparison_table(results_by_model: dict, kpis_by_model: dict) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_ORDER:
        if model_id not in results_by_model:
            continue
        k = kpis_by_model[model_id]
        rows.append(
            {
                "Modelo": MODEL_LABELS[model_id],
                "Energia [kWh]": k["energy_kWh"],
                "Pico [W]": k["p_max_W"],
                "Média [W]": k["p_mean_W"],
                "Eficiência [%]": k["eta_energy"] * 100.0,
                "PR [-]": k["PR"],
                "Tc máxima [°C]": k["tc_max_C"],
            }
        )
    return pd.DataFrame(rows)


def render_comparison_page() -> None:
    if _empty_results("Execute uma janela na seção Entrada antes de comparar os modelos."):
        return

    results = st.session_state["results_by_model"]
    kpis = st.session_state["kpis_by_model"]
    profile = st.session_state["profile"]
    module = st.session_state["module"]
    available_models = [model_id for model_id in MODEL_ORDER if model_id in results]
    preferred_reference = MODEL_SDM if MODEL_SDM in available_models else available_models[0]
    if st.session_state.get("comparison_reference_model") not in available_models:
        st.session_state["comparison_reference_model"] = preferred_reference

    energies = [kpis[mid]["energy_kWh"] for mid in results]
    peaks = [kpis[mid]["p_max_W"] for mid in results]
    energy_spread = max(energies) - min(energies)
    energy_spread_pct = energy_spread / np.mean(energies) * 100.0 if np.mean(energies) else 0.0
    peak_spread = max(peaks) - min(peaks)

    controls, main_chart = st.columns([0.62, 1.78], gap="small")
    with controls:
        with st.container(border=True):
            panel_title("Comparação · Consistência multimodelo")
            st.markdown(
                """
                <div class="model-page-title">COMPARAÇÃO DOS MODELOS</div>
                <div class="model-page-subtitle">
                  Sobreposição, dispersão e diferença relativa para a mesma janela.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True, height="stretch"):
            panel_title("Configurações · Modelo de referência")
            reference_model = st.selectbox(
                "Modelo de referência",
                options=available_models,
                format_func=lambda model_id: MODEL_LABELS[model_id],
                key="comparison_reference_model",
                help="As diferenças percentuais dos demais modelos serão calculadas em relação a esta curva.",
            )
            st.markdown(
                f"""
                <div class="config-facts">
                  <div class="config-fact"><small>Módulo</small><b>{module.stc.model}</b></div>
                  <div class="config-fact"><small>Janela</small><b>{_window_label(profile)} · passo 1 min</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            chips = []
            for model_id in MODEL_ORDER:
                available = model_id in results
                chips.append(
                    status_chip(
                        ("● " if available else "○ ") + MODEL_SHORT_LABELS[model_id],
                        "ok" if available else "warn",
                    )
                )
            st.markdown(
                '<div class="status-row">' + "".join(chips) + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="formula-box">
                  <b>Referência: {MODEL_SHORT_LABELS[reference_model]}.</b><br>
                  ΔP = (P<sub>modelo</sub> − P<sub>referência</sub>) /
                  P<sub>referência</sub> × 100.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with main_chart:
        with st.container(border=True, height="stretch"):
            panel_title("Síntese · Indicadores da comparação")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Modelos comparados", f"{len(results)}")
            q2.metric(
                "Dispersão de energia",
                f"{energy_spread:.4f} kWh",
                f"{energy_spread_pct:.2f} % da média",
                delta_color="off",
            )
            q3.metric("Dispersão de pico", f"{peak_spread:.1f} W")
            q4.metric("Janela", _window_label(profile), "1 minuto por passo", delta_color="off")
            panel_title("Curva principal · Potência · Sobreposição dos modelos")
            st.plotly_chart(
                plot_comparison_power(results, height=250),
                width="stretch",
                config=CHART_CONFIG,
                key="comparison_power",
            )

    if len(results) < 2:
        st.warning(
            "Somente um modelo está disponível. A comparação cruzada requer pelo menos dois resultados."
        )

    difference_fig = plot_difference_to_reference(results, reference_model)
    energy_col, efficiency_col, difference_col = st.columns(3, gap="small")
    with energy_col:
        with st.container(border=True, height="stretch"):
            panel_title("Energia · Acumulada")
            st.plotly_chart(
                plot_comparison_energy(results),
                width="stretch",
                config=CHART_CONFIG,
                key="comparison_energy",
            )
    with efficiency_col:
        with st.container(border=True, height="stretch"):
            panel_title("Eficiência · Sobreposição")
            st.plotly_chart(
                plot_comparison_efficiency(results),
                width="stretch",
                config=CHART_CONFIG,
                key="comparison_efficiency",
            )
    with difference_col:
        with st.container(border=True, height="stretch"):
            if difference_fig is not None:
                panel_title(
                    f"Potência · Diferença relativa a {MODEL_SHORT_LABELS[reference_model]}"
                )
                st.plotly_chart(
                    difference_fig,
                    width="stretch",
                    config=CHART_CONFIG,
                    key=f"comparison_difference_{reference_model}",
                )
            else:
                panel_title("Diferença relativa à referência")
                st.markdown(
                    '<div class="compact-note"><b>Sem outra curva disponível</b>'
                    "Execute ao menos dois modelos para calcular a diferença relativa.</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Síntese numérica · Indicadores comparáveis", expanded=False):
        table = _comparison_table(results, kpis)
        st.dataframe(
            table,
            width="stretch",
            height=160,
            hide_index=True,
            column_config={
                "Energia [kWh]": st.column_config.NumberColumn(format="%.5f"),
                "Pico [W]": st.column_config.NumberColumn(format="%.2f"),
                "Média [W]": st.column_config.NumberColumn(format="%.2f"),
                "Eficiência [%]": st.column_config.NumberColumn(format="%.3f"),
                "PR [-]": st.column_config.NumberColumn(format="%.4f"),
                "Tc máxima [°C]": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "A divergência é calculada em relação ao modelo de referência selecionado; "
            "a exatidão deve ser validada posteriormente contra potência medida."
        )


def render_export_page() -> None:
    page_header(
        "Exportação · Saída configurável",
        "CSV PARA O OTIMIZADOR E O GÊMEO DIGITAL",
        "Escolha o modelo e monte o arquivo somente com as colunas necessárias.",
    )
    if _empty_results("Execute pelo menos um modelo antes de preparar a exportação."):
        return

    results = st.session_state["results_by_model"]
    kpis = st.session_state["kpis_by_model"]
    available_models = [mid for mid in MODEL_ORDER if mid in results]

    left, right = st.columns([0.78, 1.42], gap="medium")
    with left:
        with st.container(border=True):
            panel_title("Configuração do arquivo")
            selected_model = st.selectbox(
                "Modelo exportado",
                available_models,
                format_func=lambda mid: MODEL_LABELS[mid],
            )
            selected_results = results[selected_model]
            available_columns = available_export_columns(selected_results)
            default_columns = [col for col in DEFAULT_EXPORT_COLUMNS if col in available_columns]
            selected_columns = st.multiselect(
                "Colunas incluídas",
                options=available_columns,
                default=default_columns,
                key=f"export_columns_{selected_model}",
            )
            file_name = st.text_input(
                "Nome do arquivo",
                value=f"modelo_solar_{selected_model}_{_window_slug(selected_results)}",
                key=f"export_name_{selected_model}",
            )
            separator_label = st.selectbox("Separador", ("Vírgula (,) ", "Ponto e vírgula (;)")).strip()
            decimal_label = st.selectbox("Separador decimal", ("Ponto (.)", "Vírgula (,)")).strip()
            separator = ";" if ";" in separator_label else ","
            decimal = "," if "Vírgula" in decimal_label else "."
            final_name = normalize_filename(file_name)

            st.markdown(
                '<div class="status-row">'
                + status_chip(f"{len(selected_results)} linhas", "ok")
                + status_chip(f"{len(selected_columns)} colunas", "info")
                + status_chip(MODEL_SHORT_LABELS[selected_model], "info")
                + "</div>",
                unsafe_allow_html=True,
            )

    with right:
        if not selected_columns:
            st.warning("Selecione pelo menos uma coluna para gerar o arquivo.")
            return
        export_df = build_export_dataframe(selected_results, selected_columns)
        kpi = kpis[selected_model]
        with st.container(border=True):
            panel_title("Pré-visualização · Arquivo final")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Linhas", f"{len(export_df)}")
            q2.metric("Colunas", f"{len(export_df.columns)}")
            q3.metric("Energia", f"{kpi['energy_kWh']:.4f} kWh")
            q4.metric("Pico", f"{kpi['p_max_W']:.1f} W")
            st.dataframe(export_df, width="stretch", height=340, hide_index=True)

            csv_bytes = export_df.to_csv(
                index=False,
                sep=separator,
                decimal=decimal,
                float_format="%.6f",
            ).encode("utf-8-sig")
            st.download_button(
                "⬇️ BAIXAR CSV",
                data=csv_bytes,
                file_name=final_name,
                mime="text/csv",
                type="primary",
                width="stretch",
            )


navigation = sidebar()

if navigation == NAV_OVERVIEW:
    render_overview_page()
elif navigation == NAV_INPUT:
    render_input_page()
elif navigation == NAV_MODELS:
    render_models_page()
elif navigation == NAV_COMPARISON:
    render_comparison_page()
else:
    render_export_page()
