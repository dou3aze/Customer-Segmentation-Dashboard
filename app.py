import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

# ── Palette ────────────────────────────────────────────────────────────────────
P = {
    "bg":      "#0A0A0A",
    "surface": "#1B1B1B",
    "purple":  "#AC7AF6",
    "yellow":  "#EEE59D",
    "risk":    "#712C19",
    "text":    "#F0F0F0",
    "muted":   "#7a7a8c",
}
SEGMENT_COLORS = {
    "High Income - High Spending (VIP)":   "#AC7AF6",
    "High Income - Low Spending (At Risk)": "#712C19",
    "Low Income - High Spending":           "#EEE59D",
    "Low Income - Low Spending":            "#5a4a7a",
    "Average Customers":                    "#3a3a4a",
}
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Source Serif 4', Georgia, serif", color=P["text"]),
)
AXIS = dict(gridcolor="#2a2a2a", tickfont=dict(color=P["muted"]))

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Serif+4:wght@300;400;600&display=swap');
    html, body, [class*="css"], .stApp {{ font-family: 'Source Serif 4', Georgia, serif; background-color: {P["bg"]}; color: {P["text"]}; }}
    [data-testid="stSidebar"] {{ background-color: {P["surface"]} !important; border-right: 1px solid #2a2a2a; }}
    [data-testid="stSidebar"] * {{ color: {P["text"]} !important; }}
    [data-testid="stHeader"] {{ background-color: {P["bg"]}; }}
    h1, h2, h3, h4 {{ font-family: 'Playfair Display', Georgia, serif !important; color: {P["text"]} !important; font-weight: 700 !important; }}
    p, li, span, label {{ color: #c0c0cc; }}
    strong {{ color: {P["text"]} !important; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; background: {P["surface"]}; border-radius: 10px; padding: 4px; border: 1px solid #2a2a2a; }}
    .stTabs [data-baseweb="tab"] {{ height: 40px; padding: 0 20px; border-radius: 7px; color: {P["muted"]} !important; font-size: 0.875rem; font-weight: 500; font-family: 'Source Serif 4', serif !important; }}
    .stTabs [aria-selected="true"] {{ background: {P["purple"]} !important; color: #fff !important; font-weight: 600 !important; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
    [data-testid="stMetric"] {{ background: {P["surface"]}; border: 1px solid #2a2a2a; border-left: 4px solid {P["purple"]}; border-radius: 10px; padding: 1rem 1.2rem; }}
    [data-testid="stMetricLabel"] {{ color: {P["muted"]} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ color: {P["text"]} !important; font-size: 1.75rem !important; font-weight: 700 !important; font-family: 'Playfair Display', serif !important; }}
    [data-testid="stDataFrame"] {{ border: 1px solid #2a2a2a; border-radius: 10px; overflow: hidden; }}
    [data-testid="stExpander"] {{ background: {P["surface"]}; border: 1px solid #2a2a2a !important; border-radius: 10px; }}
    [data-testid="stAlert"] {{ background: {P["surface"]} !important; border: 1px solid #2a2a2a !important; border-radius: 10px; color: {P["text"]} !important; }}
    [data-baseweb="tag"] {{ background-color: {P["purple"]} !important; color: #fff !important; border-radius: 6px !important; }}
    [data-testid="stTextInput"] input {{ background-color: {P["surface"]} !important; border-color: #2a2a2a !important; color: {P["text"]} !important; border-radius: 8px !important; }}
    hr {{ border-color: #2a2a2a !important; margin: 1.2rem 0 !important; }}
    ::-webkit-scrollbar {{ width: 5px; }} ::-webkit-scrollbar-track {{ background: {P["bg"]}; }} ::-webkit-scrollbar-thumb {{ background: #2a2a2a; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ── Data & Clustering ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/Mall_Customers.csv")
    df.columns = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled = StandardScaler().fit_transform(df[["Annual Income (k$)", "Spending Score (1-100)"]])

    # ── KMeans ──────────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = km.fit_predict(X_scaled)

    c = km.cluster_centers_
    avg_idx = int(np.argmin([np.sqrt(x**2 + y**2) for x, y in c]))
    im, sm = np.median(c[:, 0]), np.median(c[:, 1])
    def label(i, x, y):
        if i == avg_idx: return "Average Customers"
        return ("High" if x >= im else "Low") + " Income - " + ("High" if y >= sm else "Low") + " Spending" + (" (VIP)" if x >= im and y >= sm else " (At Risk)" if x >= im else "")
    df["Segment"] = df["Cluster"].map({i: label(i, *c[i]) for i in range(5)})

    # ── Elbow + Silhouette for k=2..10 ──────────────────────────────────────────
    inertias, sil_scores = [], []
    for k in range(1, 11):
        km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(km_k.inertia_)
        if k >= 2:
            sil_scores.append(silhouette_score(X_scaled, km_k.labels_))
        else:
            sil_scores.append(None)

    # Silhouette sample values for k=5 (for silhouette plot)
    sil_samples = silhouette_samples(X_scaled, df["Cluster"].values)
    df["Silhouette"] = sil_samples
    kmeans_sil = silhouette_score(X_scaled, df["Cluster"].values)

    # ── Hierarchical Clustering ──────────────────────────────────────────────────
    hc = AgglomerativeClustering(n_clusters=5, linkage="ward")
    df["HC_Cluster"] = hc.fit_predict(X_scaled)
    hc_sil = silhouette_score(X_scaled, df["HC_Cluster"].values)

    # Map HC clusters to same labels based on centroid position
    hc_centers = np.array([X_scaled[df["HC_Cluster"] == i].mean(axis=0) for i in range(5)])
    hc_avg_idx = int(np.argmin([np.sqrt(x**2 + y**2) for x, y in hc_centers]))
    hc_im, hc_sm = np.median(hc_centers[:, 0]), np.median(hc_centers[:, 1])
    def hc_label(i, x, y):
        if i == hc_avg_idx: return "Average Customers"
        return ("High" if x >= hc_im else "Low") + " Income - " + ("High" if y >= hc_sm else "Low") + " Spending" + (" (VIP)" if x >= hc_im and y >= hc_sm else " (At Risk)" if x >= hc_im else "")
    df["HC_Segment"] = df["HC_Cluster"].map({i: hc_label(i, *hc_centers[i]) for i in range(5)})

    # Dendrogram linkage matrix (sample 50 points to keep it readable)
    sample_idx = np.random.RandomState(42).choice(len(X_scaled), 50, replace=False)
    Z = linkage(X_scaled[sample_idx], method="ward")

    return df, inertias, sil_scores, kmeans_sil, hc_sil, Z

df, elbow_inertia, sil_scores, kmeans_sil, hc_sil, linkage_matrix = load_data()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Customer\nSegmentation")
st.sidebar.caption("Mall Analysis Dashboard")

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Segments", "Customers", "Recommendations", "Methodology"])

# ── TAB 1: OVERVIEW ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## Dashboard Overview")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("Avg Annual Income", f"${df['Annual Income (k$)'].mean():.1f}k")
    c3.metric("Avg Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}/100")
    c4.metric("Average Age", f"{df['Age'].mean():.1f} yrs")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Segment Distribution")
        counts = df["Segment"].value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.45,
            marker=dict(colors=list(SEGMENT_COLORS.values()), line=dict(color="#fff", width=2))
        ))
        fig.update_layout(**CHART_LAYOUT, height=380, showlegend=True,
                          xaxis=AXIS, yaxis=AXIS,
                          legend=dict(orientation="v", x=1.05, y=0.5))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Avg Income vs Spending by Segment")
        stats = df.groupby("Segment")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().reset_index()
        fig = go.Figure([
            go.Bar(name="Avg Income ($k)", x=stats["Segment"], y=stats["Annual Income (k$)"], marker_color=P["purple"]),
            go.Bar(name="Avg Spending Score", x=stats["Segment"], y=stats["Spending Score (1-100)"], marker_color=P["yellow"]),
        ])
        fig.update_layout(**CHART_LAYOUT, height=380, barmode="group", xaxis_tickangle=-35,
                          xaxis=AXIS, yaxis=AXIS,
                          legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Quick Stats")
    cols = st.columns(len(df["Segment"].unique()))
    for idx, seg in enumerate(df["Segment"].unique()):
        s = df[df["Segment"] == seg]
        with cols[idx]:
            st.markdown(f"**{seg}**")
            st.markdown(f"Count: {len(s)}  \nAvg Age: {s['Age'].mean():.1f}  \nIncome: ${s['Annual Income (k$)'].mean():.1f}k  \nSpending: {s['Spending Score (1-100)'].mean():.1f}/100")

# ── TAB 2: SEGMENTS ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## Customer Segments")
    st.markdown("---")
    st.markdown("### Income vs Spending Score")
    fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)", color="Segment",
                     hover_data=["CustomerID", "Age", "Gender"], color_discrete_map=SEGMENT_COLORS, height=480)
    fig.update_layout(**CHART_LAYOUT, xaxis=AXIS, yaxis=AXIS,
                      legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    for seg in df["Segment"].unique():
        s = df[df["Segment"] == seg]
        with st.expander(seg):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{len(s)} customers** ({len(s)/len(df)*100:.1f}%)  \n"
                            f"Avg Income: **${s['Annual Income (k$)'].mean():.1f}k**  \n"
                            f"Avg Spending: **{s['Spending Score (1-100)'].mean():.1f}/100**  \n"
                            f"Avg Age: **{s['Age'].mean():.1f} yrs**  \n"
                            f"Gender: **{(s['Gender']=='Male').sum()}M / {(s['Gender']=='Female').sum()}F**")
            with col2:
                fig = go.Figure(go.Bar(
                    x=["Min", "Avg", "Max"],
                    y=[s["Annual Income (k$)"].min(), s["Annual Income (k$)"].mean(), s["Annual Income (k$)"].max()],
                    marker_color=SEGMENT_COLORS[seg]
                ))
                fig.update_layout(**CHART_LAYOUT, height=200, showlegend=False,
                                  xaxis=AXIS, yaxis=AXIS,
                                  margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: CUSTOMERS ────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## Customer Explorer")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    search_id     = c1.text_input("Customer ID", "")
    seg_filter    = c2.multiselect("Segment", ["All"] + list(df["Segment"].unique()), default=["All"])
    gender_filter = c3.multiselect("Gender", ["All", "Male", "Female"], default=["All"])
    age_range     = c4.slider("Age", int(df["Age"].min()), int(df["Age"].max()),
                               (int(df["Age"].min()), int(df["Age"].max())))

    fdf = df.copy()
    if search_id:                              fdf = fdf[fdf["CustomerID"].astype(str).str.contains(search_id)]
    if "All" not in seg_filter and seg_filter: fdf = fdf[fdf["Segment"].isin(seg_filter)]
    if "All" not in gender_filter:             fdf = fdf[fdf["Gender"].isin(gender_filter)]
    fdf = fdf[(fdf["Age"] >= age_range[0]) & (fdf["Age"] <= age_range[1])]

    st.markdown(f"Showing **{len(fdf)}** of **{len(df)}** customers")
    out = fdf[["CustomerID", "Age", "Gender", "Annual Income (k$)", "Spending Score (1-100)", "Segment"]].copy()
    out.columns = ["ID", "Age", "Gender", "Income ($k)", "Spending Score", "Segment"]
    st.dataframe(out, use_container_width=True, hide_index=True, height=500)

# ── TAB 4: RECOMMENDATIONS ──────────────────────────────────────────────────────
with tab4:
    st.markdown("## Strategic Recommendations")
    st.markdown("---")
    recs = {
        "High Income - High Spending (VIP)":    "Offer exclusive deals, loyalty programs, and premium services.",
        "High Income - Low Spending (At Risk)":  "Use targeted promotions and personalized marketing to boost engagement.",
        "Low Income - High Spending":            "Encourage loyalty with rewards and discounts.",
        "Low Income - Low Spending":             "Low priority — use broad awareness campaigns.",
        "Average Customers":                     "Apply standard strategies and monitor for segment migration.",
    }
    for seg, rec in recs.items():
        c1, c2 = st.columns([2, 1])
        c1.markdown(f"### {seg}")
        c1.info(f"**Strategy:** {rec}")
        c2.metric("Customers", f"{len(df[df['Segment']==seg])} ({len(df[df['Segment']==seg])/len(df)*100:.1f}%)")
        st.markdown("---")

# ── TAB 5: METHODOLOGY ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("## Clustering Methodology")
    st.markdown("---")

    # ── Section 1: Elbow + Silhouette scores side by side ───────────────────────
    st.markdown("### Choosing the Optimal Number of Clusters")
    st.markdown(
        "Two complementary methods were used to justify k=5: the **Elbow Method** "
        "(inertia drops sharply then levels off) and the **Silhouette Score** "
        "(measures how well each point fits its cluster vs. the nearest other cluster — "
        "closer to 1 is better). Both converge on k=5 as the optimal choice."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Elbow Method")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)), y=elbow_inertia, mode="lines+markers",
            marker=dict(size=7, color=P["purple"]), line=dict(width=2, color=P["purple"])
        ))
        fig.add_vline(x=5, line_dash="dash", line_color=P["purple"],
                      annotation_text="k=5", annotation_position="top right")
        fig.update_layout(**CHART_LAYOUT, height=320,
                          xaxis=dict(**AXIS, title="Number of Clusters (k)", tickmode="linear", dtick=1),
                          yaxis=dict(**AXIS, title="Inertia"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Silhouette Score by k")
        sil_x = list(range(2, 11))
        sil_y = sil_scores[1:]  # skip k=1
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sil_x, y=sil_y, mode="lines+markers",
            marker=dict(size=7, color=P["yellow"]), line=dict(width=2, color=P["yellow"])
        ))
        fig.add_vline(x=5, line_dash="dash", line_color=P["purple"],
                      annotation_text="k=5", annotation_position="top right")
        fig.update_layout(**CHART_LAYOUT, height=320,
                          xaxis=dict(**AXIS, title="Number of Clusters (k)", tickmode="linear", dtick=1),
                          yaxis=dict(**AXIS, title="Silhouette Score"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Section 2: KMeans vs Hierarchical comparison ─────────────────────────────
    st.markdown("### KMeans vs. Hierarchical Clustering (k=5)")
    st.markdown(
        "To validate the KMeans results, Agglomerative Hierarchical Clustering (Ward linkage) "
        "was also applied with k=5. The scatter plots below show how each method partitions "
        "the data. A higher silhouette score indicates tighter, better-separated clusters."
    )

    # Silhouette score comparison metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("KMeans Silhouette Score", f"{kmeans_sil:.3f}")
    m2.metric("Hierarchical Silhouette Score", f"{hc_sil:.3f}")
    better = "KMeans" if kmeans_sil >= hc_sil else "Hierarchical"
    m3.metric("Better Method", better)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### KMeans Clusters")
        fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="Segment", color_discrete_map=SEGMENT_COLORS, height=400,
                         hover_data=["CustomerID", "Age"])
        fig.update_layout(**CHART_LAYOUT, xaxis=AXIS, yaxis=AXIS,
                          legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Hierarchical Clusters")
        fig = px.scatter(df, x="Annual Income (k$)", y="Spending Score (1-100)",
                         color="HC_Segment", color_discrete_map=SEGMENT_COLORS, height=400,
                         hover_data=["CustomerID", "Age"])
        fig.update_layout(**CHART_LAYOUT, xaxis=AXIS, yaxis=AXIS,
                          legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

    # Segment size comparison table
    st.markdown("#### Segment Size Comparison")
    km_counts = df["Segment"].value_counts().rename("KMeans Count")
    hc_counts = df["HC_Segment"].value_counts().rename("Hierarchical Count")
    comparison = pd.concat([km_counts, hc_counts], axis=1).fillna(0).astype(int)
    st.dataframe(comparison, use_container_width=True)

    st.markdown("---")
    st.markdown("### Dynamic Segment Labeling")
    st.markdown(
        "Labels are derived from centroid positions, not hardcoded indices. Each centroid is "
        "compared against the median income and spending across all centroids. The centroid "
        "closest to the origin (scaled space) becomes *Average Customers*. This approach "
        "applies to both KMeans and Hierarchical clustering, ensuring consistent and "
        "reproducible labels regardless of how each algorithm internally numbers its clusters."
    )