import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFERENCES = ["NeurIPS", "ICLR", "ICML", "AAAI"]
YEARS = [2026, 2025, 2024, 2023]

# Years to hide per conference (not yet released / no data)
_EXCLUDED_YEARS: Dict[str, List[int]] = {
    "NeurIPS": [2026],
    "ICML": [2026],
}

_MISSING_DATA_NOTES: Dict[str, str] = {
    "AAAI": (
        "AAAI papers on OpenReview require authors to individually opt in to public release. "
        "Run `aaai{year}/crawl_aaai.sh` to fetch what is currently available."
    ),
    "ICML": (
        "ICML {year} papers may not yet be publicly released on OpenReview. "
        "Run `icml{year}/crawl_icml.sh` after the official paper release date."
    ),
}

ANALYSIS_DIR_OVERRIDES: Dict[tuple, Path] = {}
_CLUSTER_PALETTE = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
# Visually distinct palette for up to ~20 high-level topics
_TOPIC_PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261",
    "#a8dadc", "#6a994e", "#c77dff", "#ef476f", "#06d6a0",
    "#118ab2", "#ffd166", "#ff9f1c", "#cbf3f0", "#ff6b6b",
    "#4ecdc4", "#ffe66d", "#a29bfe", "#fd79a8", "#00b894",
]


def get_analysis_dir(conference: str, year: int) -> Path:
    key = (conference, year)
    return ANALYSIS_DIR_OVERRIDES.get(key, Path(f"{conference.lower()}{year}/analysis"))


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_papers(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_embeddings(path: str) -> Optional[np.ndarray]:
    p = Path(path)
    return np.load(path) if p.exists() else None


@st.cache_data
def load_cluster_summary(path: str) -> List[dict]:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return []


@st.cache_data
def load_topics(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


@st.cache_data
def load_paper_topics(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with p.open() as f:
            data = json.load(f)
            return {item["paper_id"]: item["topics"] for item in data}
    return {}


@st.cache_data
def load_keywords(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=["keyword", "count"])
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["keyword", "count"])


@st.cache_data
def compute_pca_2d(embeddings_path: str, n_papers: int) -> Optional[np.ndarray]:
    """PCA to 2D — cached per path so it only runs once per dataset."""
    p = Path(embeddings_path)
    if not p.exists():
        return None
    emb = np.load(embeddings_path)
    return PCA(n_components=2, random_state=42).fit_transform(emb)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def cosine_similarities(vecs: np.ndarray, index: int, top_k: int = 10) -> List[int]:
    target = vecs[index]
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(target)
    scores = np.dot(vecs, target) / np.where(norms == 0, 1, norms)
    top_idx = np.argsort(scores)[::-1]
    return [i for i in top_idx if i != index][:top_k]


def inject_styles() -> None:
    st.markdown(
        """<style>
        div[data-baseweb="popover"] { z-index: 10000 !important; }
        div[data-baseweb="popover"] ul {
            background: #111827 !important;
            border: 1px solid rgba(148,163,184,0.25) !important;
        }
        div[data-baseweb="popover"] li { background: #111827 !important; }
        </style>""",
        unsafe_allow_html=True,
    )


def _scatter_fig(
    plot_df: pd.DataFrame,
    highlight_ids: Optional[Set[str]] = None,
    title: str = "",
    height: int = 560,
    color_col: str = "cluster_label",
) -> go.Figure:
    """Build a 2D scatter coloured by `color_col`.
    When coloring by topic (~13 groups) the legend is shown inside the chart.
    When coloring by cluster (50 groups) the legend is hidden (use HTML grid below)."""
    fig = go.Figure()
    by_topic = color_col != "cluster_label"
    palette = _TOPIC_PALETTE if by_topic else _CLUSTER_PALETTE

    if highlight_ids is not None:
        other = plot_df[~plot_df["paper_id"].isin(highlight_ids)]
        if not other.empty:
            fig.add_trace(go.Scatter(
                x=other["x"], y=other["y"], mode="markers",
                marker=dict(color="#999999", size=3, opacity=0.25),
                hoverinfo="skip", showlegend=False, name="other",
            ))
        sub = plot_df[plot_df["paper_id"].isin(highlight_ids)]
    else:
        sub = plot_df

    groups = sorted(sub[color_col].unique())
    color_map = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    for label, group in sub.groupby(color_col):
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers",
            marker=dict(
                color=color_map[label],
                size=5 if highlight_ids else 4,
                opacity=0.85 if highlight_ids else 0.65,
            ),
            name=label,
            showlegend=by_topic,
            customdata=group["paper_id"].values,
            text=group["short_title"].values,
            hovertemplate="<b>%{text}</b><br><i>%{fullData.name}</i><extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=title, font_size=13) if title else {},
        height=height,
        margin=dict(l=0, r=180 if by_topic else 0, t=30 if title else 10, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=by_topic,
        legend=dict(
            title="Topic", font_size=11, itemsizing="constant",
            bgcolor="rgba(128,128,128,0.12)",
            bordercolor="rgba(128,128,128,0.35)",
            borderwidth=1, orientation="v",
            x=1.0, xanchor="left", y=0.75, yanchor="top",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ) if by_topic else {},
    )
    return fig


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------

def render_overview(
    conference: str,
    year: int,
    df: pd.DataFrame,
    embeddings_path: str,
    cluster_summary: List[dict],
    topic_hierarchy: dict,
    keywords_path: str,
    paper_topics: dict,
) -> None:
    ns = f"{conference}_{year}_ov"

    for key in [f"{ns}_topic", f"{ns}_subtopic", f"{ns}_paper_id"]:
        if key not in st.session_state:
            st.session_state[key] = None

    selected_topic: Optional[str] = st.session_state[f"{ns}_topic"]
    selected_subtopic: Optional[str] = st.session_state[f"{ns}_subtopic"]

    # Pre-compute 2D coords
    coords = compute_pca_2d(embeddings_path, len(df))
    plot_df = None
    if coords is not None and len(coords) == len(df):
        plot_df = df[["paper_id", "title", "cluster_label", "cluster_id"]].copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["cluster_label"] = plot_df["cluster_label"].fillna("Unknown")
        # Add primary topic column for high-level colouring
        if paper_topics:
            plot_df["topic"] = plot_df["paper_id"].map(
                lambda pid: next(iter(paper_topics.get(str(pid), {})), "Other")
            )
        else:
            plot_df["topic"] = "Other"
        plot_df["short_title"] = plot_df["title"].fillna("").str[:80]

    # Paper IDs for selected topic
    topic_paper_ids: Optional[Set[str]] = None
    if selected_topic and paper_topics:
        topic_paper_ids = {
            pid for pid, topics in paper_topics.items() if selected_topic in topics
        }

    # Paper IDs for selected subtopic
    subtopic_paper_ids: Optional[Set[str]] = None
    if selected_topic and selected_subtopic and paper_topics:
        subtopic_paper_ids = {
            pid for pid, topics in paper_topics.items()
            if selected_topic in topics and selected_subtopic in topics.get(selected_topic, [])
        }

    # ── STAT CARDS ─────────────────────────────────────────────────────────
    top_topic = max(topic_hierarchy, key=lambda k: topic_hierarchy[k]["count"]) if topic_hierarchy else "—"
    top_count = topic_hierarchy[top_topic]["count"] if topic_hierarchy else 0
    c1, c2, c3 = st.columns([1, 2, 1])
    c1.metric("Total Papers", f"{len(df):,}")
    with c2:
        st.markdown("<p style='font-size:0.875rem;color:grey;margin-bottom:4px'>Largest Topic</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:1.5rem;font-weight:700;margin:0'>{top_topic}</p>", unsafe_allow_html=True)
    c3.metric("Papers in Topic", f"{top_count:,}", help=f"{top_count / len(df):.0%} of all papers")

    st.divider()

    # ── ROW 1: TOPICS (left) | KEYWORDS (right) — both clickable ───────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Research Topics")
        st.caption("Click a bar to drill down ↓")
        if topic_hierarchy:
            topic_df = pd.DataFrame([
                {"Topic": k, "Papers": v["count"]} for k, v in topic_hierarchy.items()
            ]).sort_values("Papers")
            bar_colors = ["#ff7f0e" if t == selected_topic else "#1f77b4" for t in topic_df["Topic"]]
            fig_t = go.Figure(go.Bar(
                x=topic_df["Papers"], y=topic_df["Topic"], orientation="h",
                marker_color=bar_colors,
                hovertemplate="%{y}: <b>%{x}</b> papers<extra></extra>",
            ))
            fig_t.update_layout(
                height=max(380, len(topic_df) * 28),
                margin=dict(l=0, r=10, t=10, b=0),
                xaxis_title="Papers", yaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            t_event = st.plotly_chart(fig_t, on_select="rerun",
                                      key=f"{ns}_topic_chart", use_container_width=True)
            if t_event.selection.points:
                clicked = t_event.selection.points[0].get("y") or t_event.selection.points[0].get("label")
                if clicked and clicked != selected_topic:
                    st.session_state[f"{ns}_topic"] = clicked
                    st.session_state[f"{ns}_subtopic"] = None
                    st.session_state[f"{ns}_paper_id"] = None
                    st.rerun()

        if selected_topic:
            cols = st.columns([4, 1])
            cols[0].success(f"**{selected_topic}** — {len(topic_paper_ids or [])} papers")
            if cols[1].button("✕", key=f"{ns}_clear", help="Clear selection"):
                st.session_state[f"{ns}_topic"] = None
                st.session_state[f"{ns}_subtopic"] = None
                st.session_state[f"{ns}_paper_id"] = None
                st.rerun()

    with right:
        kw_title = f"Top Keywords — {selected_topic}" if selected_topic else "Top Keywords"
        st.subheader(kw_title)

        if selected_topic and topic_paper_ids:
            # Compute keywords from the selected topic's papers on-the-fly
            topic_df_kw = df[df["paper_id"].isin(topic_paper_ids)]
            counter: Counter = Counter()
            for kws in topic_df_kw["keywords"]:
                if kws is None or isinstance(kws, float):
                    continue
                for kw in kws:
                    if kw and isinstance(kw, str):
                        counter[kw.lower().strip()] += 1
            if counter:
                kw_top = (
                    pd.DataFrame(counter.most_common(25), columns=["keyword", "count"])
                    .sort_values("count")
                )
            else:
                kw_top = pd.DataFrame(columns=["keyword", "count"])
        else:
            kw_df = load_keywords(keywords_path)
            kw_top = kw_df.head(25).sort_values("count") if not kw_df.empty else pd.DataFrame(columns=["keyword", "count"])

        if not kw_top.empty:
            fig_kw = go.Figure(go.Bar(
                x=kw_top["count"], y=kw_top["keyword"], orientation="h",
                marker=dict(color=kw_top["count"], colorscale="Teal"),
                hovertemplate="%{y}: <b>%{x}</b><extra></extra>",
            ))
            fig_kw.update_layout(
                height=max(380, len(kw_top) * 28),
                margin=dict(l=0, r=10, t=10, b=0),
                xaxis_title="Frequency", yaxis_title=None,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("No keyword data for this topic.")

    st.divider()

    # ── FULL-WIDTH SCATTER ──────────────────────────────────────────────────
    scatter_label = (
        f"Highlighting: {selected_topic}" +
        (f" › {selected_subtopic}" if selected_subtopic else "")
        if selected_topic else ""
    )
    active_ids = subtopic_paper_ids if subtopic_paper_ids is not None else topic_paper_ids

    st.subheader("Paper Landscape")
    st.caption("Coloured by topic · Click legend item to isolate · Double-click to restore · Click a dot to open details")
    if plot_df is not None:
        fig_s = _scatter_fig(
            plot_df, highlight_ids=active_ids, title=scatter_label,
            height=580, color_col="topic",
        )
        s_event = st.plotly_chart(fig_s, on_select="rerun",
                                  key=f"{ns}_scatter_chart", use_container_width=True)
        if s_event.selection.points:
            raw_cd = s_event.selection.points[0].get("customdata")
            if raw_cd is not None:
                pid = str(raw_cd[0]) if isinstance(raw_cd, list) else str(raw_cd)
                st.session_state[f"{ns}_paper_id"] = pid

    # Paper detail card (triggered by scatter click)
    selected_pid = st.session_state.get(f"{ns}_paper_id")
    if selected_pid:
        rows = df[df["paper_id"] == selected_pid]
        if not rows.empty:
            row = rows.iloc[0]
            with st.expander(f"📄 {row['title']}", expanded=True):
                st.write(row.get("abstract", ""))
                authors = row.get("authors")
                if authors is not None and not isinstance(authors, float) and len(authors) > 0:
                    st.caption("Authors: " + ", ".join(str(a) for a in authors))
                if row.get("cluster_label"):
                    st.caption(f"Cluster: {row['cluster_label']}")
                cols2 = st.columns(3)
                if row.get("paper_url"):
                    cols2[0].markdown(f"[Forum ↗]({row['paper_url']})")
                    pdf = str(row["paper_url"]).replace("/forum?", "/pdf?")
                    cols2[1].markdown(f"[PDF ↗]({pdf})")
                if cols2[2].button("✕ Close", key=f"{ns}_close_paper"):
                    st.session_state[f"{ns}_paper_id"] = None
                    st.rerun()

    # ── DRILL-DOWN (topic selected) ─────────────────────────────────────────
    if selected_topic:
        st.divider()
        st.subheader(f"Drill-down: {selected_topic}")

        topic_info = topic_hierarchy.get(selected_topic, {})
        subtopics = topic_info.get("subtopics", {})
        topic_papers_df = df[df["paper_id"].isin(topic_paper_ids)] if topic_paper_ids else pd.DataFrame()

        # ── Row A: Subtopics (clickable) | Cluster breakdown for topic ──────
        da1, da2 = st.columns([1, 1], gap="large")

        with da1:
            st.markdown("**Subtopics** — click to filter")
            if subtopics:
                sub_df = pd.DataFrame([
                    {"Subtopic": k, "Papers": v} for k, v in subtopics.items()
                ]).sort_values("Papers")
                sub_colors = [
                    "#ff7f0e" if s == selected_subtopic else "#e07000"
                    for s in sub_df["Subtopic"]
                ]
                fig_sub = go.Figure(go.Bar(
                    x=sub_df["Papers"], y=sub_df["Subtopic"], orientation="h",
                    marker_color=sub_colors,
                    hovertemplate="%{y}: <b>%{x}</b><extra></extra>",
                ))
                fig_sub.update_layout(
                    height=max(260, len(sub_df) * 26),
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Papers", yaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                sub_event = st.plotly_chart(fig_sub, on_select="rerun",
                                            key=f"{ns}_subtopic_chart", use_container_width=True)
                if sub_event.selection.points:
                    clicked_sub = (sub_event.selection.points[0].get("y")
                                   or sub_event.selection.points[0].get("label"))
                    if clicked_sub:
                        new_sub = None if clicked_sub == selected_subtopic else clicked_sub
                        st.session_state[f"{ns}_subtopic"] = new_sub
                        st.rerun()

                if selected_subtopic:
                    sc = st.columns([4, 1])
                    sc[0].info(f"**{selected_subtopic}** — {len(subtopic_paper_ids or [])} papers")
                    if sc[1].button("✕", key=f"{ns}_clear_sub"):
                        st.session_state[f"{ns}_subtopic"] = None
                        st.rerun()
            else:
                st.info("No subtopics data.")

        with da2:
            active_papers_df = (
                df[df["paper_id"].isin(subtopic_paper_ids)]
                if subtopic_paper_ids else topic_papers_df
            )
            label = f"Clusters in **{selected_subtopic}**" if selected_subtopic else "Cluster breakdown"
            st.markdown(label)
            if not active_papers_df.empty:
                cl_dist = (
                    active_papers_df.groupby("cluster_label").size()
                    .reset_index(name="Papers")
                    .sort_values("Papers")  # ascending for horizontal bar
                    .tail(20)              # top 20 clusters
                )
                fig_cl = go.Figure(go.Bar(
                    x=cl_dist["Papers"], y=cl_dist["cluster_label"], orientation="h",
                    marker=dict(color=cl_dist["Papers"], colorscale="Purples"),
                    hovertemplate="%{y}: <b>%{x}</b><extra></extra>",
                ))
                fig_cl.update_layout(
                    height=max(260, len(cl_dist) * 26),
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Papers", yaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_cl, use_container_width=True)

        # ── Paper list (full-width within drill-down) ───────────────────────
        list_df = df[df["paper_id"].isin(subtopic_paper_ids)] if subtopic_paper_ids else topic_papers_df
        if not list_df.empty:
            st.markdown(f"**Papers** ({len(list_df)})")
            show = list_df[["title", "cluster_label", "paper_url"]].copy()
            show["title"] = show["title"].str[:80]
            show["pdf_url"] = show["paper_url"].str.replace(
                "/forum?", "/pdf?", regex=False
            )
            st.dataframe(
                show.rename(columns={
                    "title": "Title", "cluster_label": "Cluster",
                    "paper_url": "Forum", "pdf_url": "PDF",
                }),
                hide_index=True, use_container_width=True, height=280,
                column_config={
                    "Forum": st.column_config.LinkColumn("Forum", display_text="Open ↗"),
                    "PDF": st.column_config.LinkColumn("PDF", display_text="PDF ↗"),
                },
            )

    st.divider()

    # ── CLUSTER SIZES (full width) ──────────────────────────────────────────
    st.subheader("Cluster Sizes")
    if cluster_summary:
        cs_df = (
            pd.DataFrame(cluster_summary)[["label", "size"]]
            .sort_values("size", ascending=False)
            .rename(columns={"label": "Cluster", "size": "Papers"})
        )
        fig_cs = px.bar(
            cs_df, x="Cluster", y="Papers",
            color="Papers", color_continuous_scale="Purples", height=380,
        )
        fig_cs.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title=None, xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cs, use_container_width=True)


# ---------------------------------------------------------------------------
# Browse Papers page
# ---------------------------------------------------------------------------

def render_paper_explorer(
    conference: str,
    year: int,
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray],
    cluster_summary: List[dict],
    topic_hierarchy: dict,
    paper_topics: dict,
) -> None:
    ns = f"{conference}_{year}"

    cluster_labels = (
        pd.DataFrame(cluster_summary)
        .sort_values("cluster_id")
        .assign(display=lambda d: d["cluster_id"].astype(str) + " - " + d["label"])
        .set_index("cluster_id")["display"]
        .to_dict()
        if cluster_summary else {}
    )

    st.sidebar.header("Filters")
    search_text = st.sidebar.text_input("Search (title or abstract)", key=f"{ns}_search")

    selected_topic = "All Topics"
    selected_subtopic = "All Subtopics"
    if topic_hierarchy:
        st.sidebar.subheader("Browse by Topic")
        selected_topic = st.sidebar.selectbox(
            "Topic", ["All Topics"] + sorted(topic_hierarchy.keys()), key=f"{ns}_topic"
        )
        if selected_topic != "All Topics":
            selected_subtopic = st.sidebar.selectbox(
                "Subtopic",
                ["All Subtopics"] + sorted(topic_hierarchy[selected_topic]["subtopics"].keys()),
                key=f"{ns}_subtopic",
            )
            st.sidebar.info(f"**{selected_topic}**: {topic_hierarchy[selected_topic]['count']} papers")
            if selected_subtopic != "All Subtopics":
                st.sidebar.info(f"**{selected_subtopic}**: {topic_hierarchy[selected_topic]['subtopics'][selected_subtopic]} papers")

    st.sidebar.subheader("Browse by Cluster")
    selected_clusters = st.sidebar.multiselect(
        "Cluster",
        options=sorted(df["cluster_id"].dropna().unique().tolist()),
        format_func=lambda c: cluster_labels.get(c, str(c)),
        key=f"{ns}_clusters",
    )

    filtered = df.copy()
    if search_text:
        lowered = search_text.lower()
        filtered = filtered[
            filtered["title"].fillna("").str.lower().str.contains(lowered)
            | filtered["abstract"].fillna("").str.lower().str.contains(lowered)
        ]
    if selected_topic != "All Topics" and paper_topics:
        matching_ids = [
            pid for pid, topics in paper_topics.items()
            if selected_topic in topics and (
                selected_subtopic == "All Subtopics" or selected_subtopic in topics[selected_topic]
            )
        ]
        filtered = filtered[filtered["paper_id"].isin(matching_ids)]
    if selected_clusters:
        filtered = filtered[filtered["cluster_id"].isin(selected_clusters)]

    st.write(f"Showing **{len(filtered)}** papers")
    if filtered.empty:
        st.info("No papers match the current filters.")
        return

    paper_ids = filtered["paper_id"].tolist()
    paper_title_map = filtered.set_index("paper_id")["title"].fillna("Untitled").to_dict()

    anchor_key = f"{ns}_anchor_paper_id"
    history_key = f"{ns}_click_history"

    if history_key not in st.session_state or (
        st.session_state[history_key]
        and not isinstance(st.session_state[history_key][0], dict)
    ):
        st.session_state[history_key] = []
    if anchor_key not in st.session_state or st.session_state[anchor_key] not in paper_ids:
        st.session_state[anchor_key] = paper_ids[0]

    anchor_id = st.session_state[anchor_key]
    paper_options = paper_ids[paper_ids.index(anchor_id):]

    selected_paper_id = st.selectbox(
        "Select a paper", options=paper_options, index=0,
        format_func=lambda pid: paper_title_map.get(pid, "Untitled"),
        key=f"{ns}_paper_selectbox",
    )
    if selected_paper_id != st.session_state[anchor_key]:
        st.session_state[anchor_key] = selected_paper_id
        st.session_state[history_key].append(
            {"paper_id": str(selected_paper_id), "title": paper_title_map.get(selected_paper_id, "")}
        )
        st.session_state[history_key] = st.session_state[history_key][-10:]
        st.rerun()

    row = filtered[filtered["paper_id"] == selected_paper_id].iloc[0]
    st.subheader(row["title"])
    st.write(row.get("abstract", ""))

    if paper_topics and row["paper_id"] in paper_topics:
        for topic, subtopics in paper_topics[row["paper_id"]].items():
            st.caption(f"  • {topic}: {', '.join(subtopics)}")

    authors = row.get("authors")
    if authors is not None and not isinstance(authors, float) and len(authors) > 0:
        st.caption("Authors: " + ", ".join(str(a) for a in authors))
    keywords = row.get("keywords")
    if keywords is not None and not isinstance(keywords, float) and len(keywords) > 0:
        st.caption("Keywords: " + ", ".join(str(k) for k in keywords))
    link_cols = st.columns(2)
    if row.get("paper_url"):
        link_cols[0].markdown(f"[Forum ↗]({row['paper_url']})")
        pdf = str(row["paper_url"]).replace("/forum?", "/pdf?")
        link_cols[1].markdown(f"[PDF ↗]({pdf})")
    if row.get("cluster_label"):
        st.caption(f"Cluster: {row['cluster_label']}")

    if embeddings is not None and len(embeddings) == len(df):
        st.subheader("Related papers")
        similar_idx = cosine_similarities(embeddings, int(row.name), top_k=10)
        rel = df.iloc[similar_idx][["title", "paper_url", "cluster_label"]].copy().reset_index(drop=True)
        rel["pdf_url"] = rel["paper_url"].str.replace("/forum?", "/pdf?", regex=False)
        st.dataframe(
            rel.rename(columns={"title": "Title", "cluster_label": "Cluster", "paper_url": "Forum", "pdf_url": "PDF"}),
            use_container_width=True,
            column_config={
                "Forum": st.column_config.LinkColumn("Forum", display_text="Open ↗"),
                "PDF": st.column_config.LinkColumn("PDF", display_text="PDF ↗"),
            },
        )
    else:
        st.info("Embeddings not found or mismatch; related-paper view is disabled.")

    st.subheader("Recently browsed papers")
    history = st.session_state.get(history_key, [])
    if history:
        st.dataframe(
            pd.DataFrame(history[::-1]).rename(columns={"paper_id": "ID", "title": "Title"}).head(10),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No browsed papers yet.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="ML Paper Explorer", layout="wide")
    inject_styles()

    st.sidebar.title("ML Paper Explorer")
    conference = st.sidebar.selectbox("Conference", CONFERENCES)
    available_years = [y for y in YEARS if y not in _EXCLUDED_YEARS.get(conference, [])]
    year = st.sidebar.selectbox("Year", available_years)
    st.sidebar.divider()
    mode = st.sidebar.radio("View", ["Overview", "Browse Papers"])

    st.title(f"{conference} {year}")

    base = get_analysis_dir(conference, year)
    data_path       = base / "papers_with_clusters.parquet"
    embeddings_path = base / "embeddings.npy"
    clusters_path   = base / "cluster_summary.json"
    topics_path     = base / "topic_hierarchy.json"
    paper_topics_path = base / "paper_topics.json"
    keywords_path   = base / "top_keywords.csv"

    if not data_path.exists():
        extra = _MISSING_DATA_NOTES.get(conference, "")
        if extra:
            extra = "\n\n" + extra.format(year=year)
        st.warning(
            f"No data found for **{conference} {year}**.\n\n"
            f"Expected: `{data_path}`\n\n"
            f"Run the crawl + analysis pipeline first, then refresh.{extra}"
        )
        st.sidebar.info(f"Data not available for {conference} {year}.")
        return

    df             = load_papers(str(data_path))
    cluster_summary = load_cluster_summary(str(clusters_path))
    topic_hierarchy = load_topics(str(topics_path))

    if mode == "Overview":
        paper_topics = load_paper_topics(str(paper_topics_path))
        render_overview(
            conference=conference, year=year, df=df,
            embeddings_path=str(embeddings_path),
            cluster_summary=cluster_summary,
            topic_hierarchy=topic_hierarchy,
            keywords_path=str(keywords_path),
            paper_topics=paper_topics,
        )
    else:
        embeddings   = load_embeddings(str(embeddings_path))
        paper_topics = load_paper_topics(str(paper_topics_path))
        render_paper_explorer(
            conference=conference, year=year, df=df,
            embeddings=embeddings,
            cluster_summary=cluster_summary,
            topic_hierarchy=topic_hierarchy,
            paper_topics=paper_topics,
        )


if __name__ == "__main__":
    main()
