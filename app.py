"""Single-command dashboard for the sentiment classification pipeline.

    streamlit run app.py

Shows the pipeline's live state (env check -> split -> each experiment's
smoke test/training/eval -> comparison -> one-time test report ->
conclusions), tails the active log file, and charts results as they land,
all driven by `results/run_state.json` (written by scripts/run_all.py) plus
the result files it produces. Starting the whole run is a button in this
app, so `streamlit run app.py` really is the one command needed end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"
PID_FILE = RESULTS_DIR / ".pipeline.pid"
PIPELINE_LOG = LOGS_DIR / "pipeline.log"

STAGE_ORDER = [
    ("env_check", "بررسی محیط اجرا"),
    ("split", "ساخت/بارگذاری Split"),
    ("experiments", "اجرای آزمایش‌ها"),
    ("compare", "مقایسه نتایج"),
    ("final_report", "گزارش نهایی روی Test"),
    ("conclusions", "نتیجه‌گیری خودکار"),
    ("done", "پایان"),
]

STATUS_COLORS = {
    "pending": "#9aa0a6",
    "smoke_test": "#f6c343",
    "training": "#4285f4",
    "evaluating": "#4285f4",
    "done": "#34a853",
    "skipped": "#7b8794",
    "failed": "#ea4335",
}

st.set_page_config(page_title="Tweet Sentiment Pipeline", layout="wide")


def load_state() -> Optional[dict]:
    path = RESULTS_DIR / "run_state.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_pipeline_running() -> bool:
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return False
    if not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        return "run_all.py" in " ".join(proc.cmdline())
    except psutil.Error:
        return False


def start_pipeline(force: bool, skip_smoke: bool) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(ROOT / "scripts" / "run_all.py")]
    if force:
        args.append("--force")
    if skip_smoke:
        args.append("--skip-smoke")
    log_file = open(PIPELINE_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        args, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")


def tail_file(path: Path, n_lines: int = 250) -> str:
    if not path.exists():
        return "(no log yet)"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(log unreadable)"
    return "\n".join(lines[-n_lines:])


def render_state_diagram(state: Optional[dict]) -> None:
    current_stage = state["stage"] if state else "env_check"
    stage_index = {key: i for i, (key, _) in enumerate(STAGE_ORDER)}
    current_i = stage_index.get(current_stage, 0)

    boxes = []
    for i, (key, label) in enumerate(STAGE_ORDER):
        if i < current_i:
            color = STATUS_COLORS["done"]
        elif i == current_i:
            color = STATUS_COLORS["failed"] if state and state.get("error") else STATUS_COLORS["training"]
        else:
            color = STATUS_COLORS["pending"]
        boxes.append(
            f'<div style="background:{color};color:white;padding:10px 16px;'
            f'border-radius:8px;margin-right:8px;white-space:nowrap;font-size:13px;">{label}</div>'
        )
    arrow = '<div style="padding:0 4px;color:#888;">→</div>'
    html = '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">' + arrow.join(boxes) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    if state and state.get("stage_detail"):
        st.caption(state["stage_detail"])
    if state and state.get("error"):
        st.error(state["error"])


def render_experiment_grid(state: Optional[dict]) -> None:
    if not state or not state.get("experiments"):
        st.info("هنوز آزمایشی صف نشده است.")
        return
    cols = st.columns(3)
    for i, (name, record) in enumerate(state["experiments"].items()):
        status = record.get("status", "pending")
        color = STATUS_COLORS.get(status, "#9aa0a6")
        epoch = record.get("epoch")
        total_epochs = record.get("total_epochs")
        epoch_text = f" — epoch {epoch}/{total_epochs}" if epoch else ""
        metrics = record.get("latest_metrics") or {}
        metric_text = ""
        if metrics.get("f1") is not None:
            metric_text = f"<br/><small>f1={metrics['f1']:.4f}</small>"
        with cols[i % 3]:
            st.markdown(
                f'<div style="border-left:5px solid {color};padding:8px 12px;'
                f'margin-bottom:8px;background:rgba(127,127,127,0.08);border-radius:4px;">'
                f"<b>{name}</b><br/><small>{status}{epoch_text}</small>{metric_text}"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_metrics_chart() -> None:
    csv_path = RESULTS_DIR / "comparison_table.csv"
    if not csv_path.exists():
        st.info("هنوز جدول مقایسه‌ای تولید نشده است.")
        return
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1"] if c in df.columns]
    fig = go.Figure()
    for metric in metric_cols:
        fig.add_trace(go.Bar(name=metric, x=df["experiment"], y=df[metric]))
    fig.update_layout(barmode="group", yaxis_title="score", xaxis_title="experiment", height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)


def render_hardware_chart(experiment_name: str) -> None:
    log_path = RESULTS_DIR / experiment_name / "hardware_log.csv"
    if not log_path.exists():
        st.info("لاگ سخت‌افزار برای این آزمایش موجود نیست.")
        return
    df = pd.read_csv(log_path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["elapsed_seconds"], y=df["cpu_percent"], name="CPU %"))
    fig.add_trace(go.Scatter(x=df["elapsed_seconds"], y=df["ram_percent"], name="RAM %"))
    if df["gpu_util_percent"].notna().any():
        fig.add_trace(go.Scatter(x=df["elapsed_seconds"], y=df["gpu_util_percent"], name="GPU %"))
    fig.update_layout(xaxis_title="seconds", yaxis_title="%", height=380)
    st.plotly_chart(fig, use_container_width=True)


st.title("Tweet Sentiment Classification — Pipeline Dashboard")

with st.sidebar:
    st.header("کنترل اجرا")
    force = st.checkbox("اجرای دوباره آزمایش‌های تکمیل‌شده (--force)", value=False)
    skip_smoke = st.checkbox("رد کردن تست دود (--skip-smoke)", value=False)
    running = is_pipeline_running()
    if running:
        st.success("پایپ‌لاین در حال اجراست...")
        if st.button("توقف (kill)"):
            try:
                psutil.Process(int(PID_FILE.read_text().strip())).terminate()
            except (psutil.Error, ValueError, OSError):
                pass
    else:
        if st.button("شروع اجرای کامل پایپ‌لاین", type="primary"):
            start_pipeline(force=force, skip_smoke=skip_smoke)
            st.rerun()

    auto_refresh = st.checkbox("رفرش خودکار (هر ۲ ثانیه)", value=running)

state = load_state()

st.subheader("جریان اجرا")
render_state_diagram(state)

tab_experiments, tab_logs, tab_metrics, tab_hardware, tab_conclusions = st.tabs(
    ["آزمایش‌ها", "لاگ زنده", "مقایسه نتایج", "مصرف سخت‌افزار", "نتیجه‌گیری"]
)

with tab_experiments:
    render_experiment_grid(state)

with tab_logs:
    st.text_area("pipeline.log", tail_file(PIPELINE_LOG), height=350)
    if state and state.get("experiments"):
        selected = st.selectbox("لاگ یک آزمایش خاص", list(state["experiments"].keys()))
        if selected:
            st.text_area(
                f"{selected}/train.log",
                tail_file(RESULTS_DIR / selected / "train.log"),
                height=350,
            )

with tab_metrics:
    render_metrics_chart()

with tab_hardware:
    if state and state.get("experiments"):
        selected_hw = st.selectbox("انتخاب آزمایش", list(state["experiments"].keys()), key="hw_select")
        if selected_hw:
            render_hardware_chart(selected_hw)
    else:
        st.info("هنوز آزمایشی اجرا نشده است.")

with tab_conclusions:
    findings_path = RESULTS_DIR / "findings.md"
    final_report_path = RESULTS_DIR / "final_test_report.md"
    if findings_path.exists():
        st.markdown(findings_path.read_text(encoding="utf-8"))
    else:
        st.info("هنوز findings.md تولید نشده است.")
    if final_report_path.exists():
        st.divider()
        st.markdown(final_report_path.read_text(encoding="utf-8"))

if auto_refresh:
    time.sleep(2)
    st.rerun()
