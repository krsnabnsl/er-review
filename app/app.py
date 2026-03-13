"""ER Review — Mission Payload Image Review Application"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ER Review",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
TABLE = os.getenv("ER_TABLE", "roboticscatalog.er_review.er_readings")


CSS = """
<style>
.badge-reviewed { background: linear-gradient(135deg,#00c853,#00897b);
                  color:#fff; padding:4px 12px; border-radius:20px;
                  font-size:.75rem; font-weight:700;
                  display:inline-block; margin-bottom:6px;
                  box-shadow: 0 2px 6px rgba(0,200,83,.35); }
.badge-pending  { background: linear-gradient(135deg,#ff6d00,#e65100);
                  color:#fff; padding:4px 12px; border-radius:20px;
                  font-size:.75rem; font-weight:700;
                  display:inline-block; margin-bottom:6px;
                  box-shadow: 0 2px 6px rgba(230,81,0,.35); }
.mission-tile   { border:1px solid #2e2e3e; border-radius:14px;
                  padding:22px 18px; text-align:center;
                  background: linear-gradient(160deg,#1a1a2e 60%,#16213e);
                  margin-bottom:6px;
                  box-shadow: 0 4px 14px rgba(0,0,0,.4);
                  transition: transform .15s; }
.mission-tile:hover { transform: translateY(-2px); }
.tile-title     { font-size:1.2rem; font-weight:800; margin-bottom:8px;
                  letter-spacing:.02em; }
.tile-sub       { color:#90caf9; font-size:.82rem; }
.section-header { font-size:1.05rem; font-weight:700; color:#90caf9;
                  text-transform:uppercase; letter-spacing:.08em;
                  margin-bottom:4px; }
</style>
"""

# ──────────────────────────────────────────────────────────────────────────────
# SPARK / DATABRICKS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_spark():
    from databricks.connect import DatabricksSession
    return DatabricksSession.builder.serverless(True).getOrCreate()


def _is_session_expired(e: Exception) -> bool:
    msg = str(e)
    return "INACTIVITY_TIMEOUT" in msg or "session_id is no longer usable" in msg


def _reset_spark_caches() -> None:
    get_spark.clear()
    load_data.clear()
    load_image_from_volume.clear()


# ──────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load er_readings from the Databricks Delta table."""
    from pyspark.sql import functions as F
    df = get_spark().table(TABLE).orderBy(F.col("er_name")).limit(5000).toPandas()
    df["date"] = pd.to_datetime(df["date"])
    df["day"]  = df["date"].dt.strftime("%Y-%m-%d")
    return df


@st.cache_data(show_spinner=False)
def load_image_from_volume(path: str) -> Image.Image | None:
    """Load an image from a Databricks Unity Catalog volume path."""
    try:
        row = get_spark().read.format("binaryFile").load(path).first()
        return ImageOps.exif_transpose(Image.open(io.BytesIO(bytes(row["content"]))))
    except Exception:
        return None


def _make_gauge_placeholder(er_name: str, er_value: float, er_angle: float) -> Image.Image:
    """Generate a synthetic gauge-style placeholder image."""
    W, H = 400, 300
    img  = Image.new("RGB", (W, H), color=(20, 22, 36))
    draw = ImageDraw.Draw(img)
    cx, cy, r = W // 2, H // 2 + 10, 100
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(100, 110, 140), width=3)
    draw.ellipse([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8], outline=(50, 55, 80), width=1)
    for deg in range(0, 360, 30):
        rad = np.radians(deg)
        x1 = int(cx + (r - 2)  * np.cos(rad))
        y1 = int(cy + (r - 2)  * np.sin(rad))
        x2 = int(cx + (r - 14) * np.cos(rad))
        y2 = int(cy + (r - 14) * np.sin(rad))
        draw.line([x1, y1, x2, y2], fill=(150, 155, 180), width=2)
    rad = np.radians(er_angle - 90)
    nx  = int(cx + (r - 18) * np.cos(rad))
    ny  = int(cy + (r - 18) * np.sin(rad))
    draw.line([cx, cy, nx, ny], fill=(255, 80, 60), width=3)
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 80, 60))
    draw.text((8, 6),      er_name,               fill=(180, 185, 210))
    draw.text((8, H - 22), f"Value: {er_value}",  fill=(180, 185, 210))
    return img


def get_image(path: str, er_name: str, er_value: float, er_angle: float) -> Image.Image:
    """Return the best available image: local file → volume → placeholder."""
    try:
        if Path(path).exists():
            return Image.open(path)
    except Exception:
        pass
    img = load_image_from_volume(path)
    return img if img is not None else _make_gauge_placeholder(er_name, er_value, er_angle)


def _persist_review(er_name: str, photo_path: str, reviewer_value: float | None) -> None:
    """Write reviewer_value back to the Delta table (no-op in mock mode)."""
    try:
        val_sql = str(reviewer_value) if reviewer_value is not None else "NULL"
        q = (
            f"UPDATE {TABLE} "
            f"SET reviewer_value = {val_sql} "
            f"WHERE er_name = '{er_name.replace(chr(39), chr(39)*2)}' "
            f"AND photo_volume_path = '{photo_path.replace(chr(39), chr(39)*2)}'"
        )
        spark = get_spark()
        spark.sql(q)
        load_data.clear()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    defaults: dict = {
        "view":                 "mission_browser",
        "df":                   None,
        "selected_mission":     None,
        "selected_payload_idx": None,
        "filter_site":          None,
        "filter_day":           None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if st.session_state.df is None:
        try:
            st.session_state.df = load_data()
        except Exception as e:
            if _is_session_expired(e):
                _reset_spark_caches()
                st.rerun()
            st.error(f"Failed to load data: {e}")
            st.stop()


def navigate(view: str, **kwargs) -> None:
    st.session_state.view = view
    for k, v in kwargs.items():
        st.session_state[k] = v


# ──────────────────────────────────────────────────────────────────────────────
# SHARED: FILTER PANEL
# ──────────────────────────────────────────────────────────────────────────────

def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render site + date slicers. Returns the filtered DataFrame."""
    col1, col2 = st.columns(2)

    # ── Site ──
    sites = sorted(df["site"].astype(str).unique().tolist())
    with col1:
        site = st.selectbox(
            "📍 Site",
            sites,
            index=sites.index(st.session_state.filter_site)
            if st.session_state.filter_site in sites else 0,
            key="_f_site",
        )
        st.session_state.filter_site = site

    df_s = df[df["site"].astype(str) == site]

    # ── Date (string "YYYY-MM-DD", already sorted) ──
    days = sorted(df_s["day"].astype(str).unique().tolist())
    with col2:
        day = st.selectbox(
            "📅 Date",
            days,
            index=days.index(str(st.session_state.filter_day))
            if str(st.session_state.filter_day) in days else 0,
            key="_f_day",
        )
        st.session_state.filter_day = day

    return df_s[df_s["day"].astype(str) == day]


# ──────────────────────────────────────────────────────────────────────────────
# VIEW 1 — MISSION BROWSER
# ──────────────────────────────────────────────────────────────────────────────

def render_mission_browser() -> None:
    st.markdown("# 🤖 ER Review")
    st.caption("Energy Robotics — payload inspection & reviewer portal")
    st.markdown("---")

    st.markdown('<p class="section-header">🎛️ Filter Panel</p>', unsafe_allow_html=True)
    filtered = render_filters(st.session_state.df)

    st.markdown("---")
    st.markdown('<p class="section-header">🦾 Missions</p>', unsafe_allow_html=True)

    missions = filtered["mission_id"].unique().tolist()
    if not missions:
        st.info("🔍 No missions found for the selected filters.")
        return

    COLS = 4
    cols = st.columns(COLS)
    for i, mission in enumerate(missions):
        m_df     = filtered[filtered["mission_id"] == mission]
        total    = len(m_df)
        reviewed = int(m_df["reviewer_value"].notna().sum())
        pct      = int(reviewed / total * 100) if total else 0

        with cols[i % COLS]:
            st.markdown(
                f"""<div class="mission-tile">
                    <div class="tile-title">⚙️ {mission}</div>
                    <div class="tile-sub">🎯 {total} payloads &nbsp;·&nbsp; ✅ {reviewed} reviewed &nbsp;({pct}%)</div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"▶ Open {mission}", key=f"btn_mission_{mission}", use_container_width=True):
                navigate("payload_gallery", selected_mission=mission)
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# VIEW 2 — PAYLOAD GALLERY
# ──────────────────────────────────────────────────────────────────────────────

def render_payload_gallery() -> None:
    df      = st.session_state.df
    mission = st.session_state.selected_mission

    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("⬅ Missions", key="gal_back"):
            navigate("mission_browser")
            st.rerun()
    with col_title:
        st.markdown(f"## ⚙️ Mission: {mission}")
        st.caption(f"📍 {st.session_state.filter_site}  ·  📅 {st.session_state.filter_day}")

    st.markdown("---")

    # filter by the active site + date so gallery matches the browser selection
    site = st.session_state.filter_site
    day  = str(st.session_state.filter_day)
    m_df = df[
        (df["mission_id"] == mission) &
        (df["site"].astype(str) == site) &
        (df["day"].astype(str) == day)
    ].copy()
    m_df["orig_idx"] = m_df.index
    m_df = m_df.reset_index(drop=True)

    if m_df.empty:
        st.warning("No payloads found for this mission.")
        return

    total    = len(m_df)
    reviewed = int(m_df["reviewer_value"].notna().sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("📸 Total Payloads", total)
    c2.metric("✅ Reviewed",       reviewed)
    c3.metric("⏳ Pending",        total - reviewed)
    st.progress(reviewed / total if total else 0)

    st.markdown("---")

    COLS = 4
    for row_start in range(0, len(m_df), COLS):
        chunk = m_df.iloc[row_start: row_start + COLS]
        cols  = st.columns(COLS)
        for col_pos, (_, payload) in enumerate(chunk.iterrows()):
            orig_idx = int(payload["orig_idx"])
            with cols[col_pos]:
                img = get_image(
                    payload["photo_volume_path"],
                    payload["er_name"],
                    float(payload["er_value"]),
                    float(payload["er_angle"]),
                )
                st.image(img, use_column_width=True)

                is_reviewed = pd.notna(payload["reviewer_value"])
                badge = (
                    '<span class="badge-reviewed">✅ Reviewed</span>'
                    if is_reviewed else
                    '<span class="badge-pending">⏳ Pending</span>'
                )
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(
                    f"**🔩 {payload['er_name']}**  \n"
                    f"📊 ER Value: `{payload['er_value']}`  \n"
                    f"🦾 Conf: `{payload['er_confidence_score']:.3f}`"
                )
                if st.button("🔍 Review", key=f"open_payload_{orig_idx}", use_container_width=True):
                    navigate("image_review", selected_payload_idx=orig_idx)
                    st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# VIEW 3 — IMAGE REVIEW
# ──────────────────────────────────────────────────────────────────────────────

def render_image_review() -> None:
    df  = st.session_state.df
    idx = st.session_state.selected_payload_idx

    if idx is None or idx not in df.index:
        st.error("No payload selected.")
        return

    payload = df.loc[idx]

    col_back, col_title = st.columns([1, 9])
    with col_back:
        if st.button("⬅ Gallery", key="rev_back"):
            navigate("payload_gallery")
            st.rerun()
    with col_title:
        st.markdown(f"## 🔍 Review: {payload['er_name']}")
        st.caption(f"⚙️ {payload['mission_id']}  ·  📍 {payload['site']}  ·  📅 {payload['date'].date()}")

    st.markdown("---")

    left, right = st.columns([3, 2], gap="large")

    # ── Left: image preview ──
    with left:
        st.markdown("#### 📷 Image Preview")
        try:
            img = get_image(
                payload["photo_volume_path"],
                payload["er_name"],
                float(payload["er_value"]),
                float(payload["er_angle"]),
            )
            st.image(img, use_column_width=True)
        except Exception as e:
            if _is_session_expired(e):
                _reset_spark_caches()
                st.rerun()
            st.error(f"Cannot load image: {e}")
        st.caption(payload["photo_volume_path"])

    # ── Right: metadata + review ──
    with right:
        st.markdown("#### 📋 Payload Metadata")
        st.markdown(
            f"| Field | Value |\n|---|---|\n"
            f"| 🏷️ **ER Name** | {payload['er_name']} |\n"
            f"| 📊 **ER Value** | {payload['er_value']} |\n"
            f"| 🧭 **ER Angle** | {payload['er_angle']}° |\n"
            f"| 🦾 **Confidence** | {payload['er_confidence_score']:.3f} |\n"
            f"| ⚙️ **Mission** | {payload['mission_id']} |\n"
            f"| 📍 **Site** | {payload['site']} |\n"
            f"| 📅 **Date** | {payload['date'].date()} |\n"
        )

        st.markdown("---")
        st.markdown("#### ✏️ Review")

        current_val = payload["reviewer_value"]
        is_reviewed = pd.notna(current_val)

        if is_reviewed:
            st.success(f"✅ Reviewed — current value: **{current_val}**")
        else:
            st.warning("⏳ Not yet reviewed")

        new_val = st.number_input(
            "📝 Reviewer Value",
            value=float(current_val) if is_reviewed else float(payload["er_value"]),
            step=0.01,
            format="%.2f",
            key=f"rev_input_{idx}",
        )

        col_save, col_unreview = st.columns(2)
        with col_save:
            if st.button("✅ Mark Reviewed", use_container_width=True, type="primary"):
                st.session_state.df.at[idx, "reviewer_value"] = new_val
                _persist_review(payload["er_name"], payload["photo_volume_path"], new_val)
                st.success("💾 Saved successfully!")
                navigate("payload_gallery")
                st.rerun()

        with col_unreview:
            if st.button("↩️ Unreview", use_container_width=True, disabled=not is_reviewed):
                st.session_state.df.at[idx, "reviewer_value"] = None
                _persist_review(payload["er_name"], payload["photo_volume_path"], None)
                st.info("Marked as unreviewed.")
                navigate("payload_gallery")
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_session_state()

    view = st.session_state.view
    if view == "mission_browser":
        render_mission_browser()
    elif view == "payload_gallery":
        render_payload_gallery()
    elif view == "image_review":
        render_image_review()
    else:
        navigate("mission_browser")
        st.rerun()


main()
