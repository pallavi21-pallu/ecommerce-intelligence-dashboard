import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# DARK THEME / CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(circle at 85% 0%, rgba(76, 110, 245, 0.12), transparent 28%),
          #0b1020;
        color: #edf2ff;
    }
    [data-testid="stSidebar"] {
        background: #10182b;
        border-right: 1px solid #24304a;
    }
    [data-testid="stSidebar"] * {
        color: #e8edff;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }
    .hero {
        padding: 1.35rem 1.55rem;
        border: 1px solid #293653;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(28,42,72,.98), rgba(17,26,46,.94));
        margin-bottom: 1.2rem;
    }
    .hero-kicker {
        color: #91a8ff;
        font-size: .78rem;
        letter-spacing: .14em;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }
    .hero h1 {
        color: #f6f8ff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    .hero p {
        color: #aebbd5;
        margin: .6rem 0 0 0;
        font-size: .96rem;
    }
    .metric-card {
        background: linear-gradient(145deg, #151f35, #111a2d);
        border: 1px solid #2b3855;
        border-radius: 16px;
        padding: 1.05rem 1.1rem;
        min-height: 118px;
        box-shadow: 0 8px 22px rgba(0,0,0,.12);
    }
    .metric-label {
        color: #9eacc8;
        font-size: .82rem;
        font-weight: 600;
        margin-bottom: .65rem;
    }
    .metric-value {
        color: #f4f7ff;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-note {
        color: #7f91b4;
        font-size: .76rem;
        margin-top: .45rem;
    }
    .section-title {
        font-size: 1.18rem;
        font-weight: 750;
        color: #f1f5ff;
        margin: .45rem 0 .8rem 0;
    }
    .small-muted {
        color: #9eacc8;
        font-size: .88rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #2b3855;
        border-radius: 12px;
        overflow: hidden;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #dce5ff !important;
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #5068bb;
        background: #263b75;
        color: white;
        font-weight: 650;
    }
    div.stButton > button:hover {
        border-color: #91a8ff;
        background: #304b91;
        color: white;
    }
    hr { border-color: #273451; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Consistent Plotly styling
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#dce5ff"),
    margin=dict(l=15, r=15, t=55, b=15),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ------------------------------------------------------------
# DATA GENERATION (cached so the dataset stays consistent)
# ------------------------------------------------------------
@st.cache_data
def create_dataset(n_transactions=10_000, seed=42):
    rng = np.random.default_rng(seed)

    products = [
        "Laptop", "Smartphone", "Headphones", "Smartwatch", "Tablet",
        "Keyboard", "Mouse", "Monitor", "Camera", "Speaker"
    ]
    categories = {
        "Laptop": "Electronics",
        "Smartphone": "Electronics",
        "Headphones": "Accessories",
        "Smartwatch": "Wearables",
        "Tablet": "Electronics",
        "Keyboard": "Accessories",
        "Mouse": "Accessories",
        "Monitor": "Electronics",
        "Camera": "Electronics",
        "Speaker": "Audio",
    }
    countries = ["India", "USA", "UK", "Germany", "Canada", "Australia"]

    customer_ids = rng.integers(1001, 2001, n_transactions)
    product_data = rng.choice(products, n_transactions)
    quantity = rng.integers(1, 6, n_transactions)
    unit_price = rng.uniform(20, 1500, n_transactions)
    purchase_dates = pd.date_range(
        start="2025-01-01", end="2025-12-31", periods=n_transactions
    )
    gender_map = {
        int(customer): rng.choice(["Male", "Female"], p=[0.55, 0.45])
        for customer in np.unique(customer_ids)
    }

    data = pd.DataFrame({
        "Customer_ID": customer_ids,
        "Product": product_data,
        "Category": [categories[p] for p in product_data],
        "Country": rng.choice(countries, n_transactions),
        "Quantity": quantity,
        "UnitPrice": unit_price,
        "PurchaseDate": purchase_dates,
    })
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data["Gender"] = data["Customer_ID"].map(gender_map)
    data["Year"] = data["PurchaseDate"].dt.year
    data["Month"] = data["PurchaseDate"].dt.month
    data["MonthName"] = data["PurchaseDate"].dt.strftime("%B")
    data["Day"] = data["PurchaseDate"].dt.day
    data["Weekday"] = data["PurchaseDate"].dt.day_name()
    return data.drop_duplicates().reset_index(drop=True)


@st.cache_data
def create_customer_analysis(data):
    reference_date = data["PurchaseDate"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("Customer_ID").agg(
        Recency=("PurchaseDate", lambda x: (reference_date - x.max()).days),
        Frequency=("Customer_ID", "count"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    # Rank first to avoid qcut failures when values have ties.
    rfm["R_Score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]
    ).astype(int)
    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    def assign_segment(row):
        if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
            return "Champions"
        if row["R_Score"] >= 4 and row["F_Score"] >= 3:
            return "Loyal Customers"
        if row["R_Score"] >= 4 and row["M_Score"] >= 3:
            return "Potential Loyalists"
        if row["R_Score"] <= 2 and row["F_Score"] >= 3:
            return "At Risk"
        if row["R_Score"] <= 2 and row["M_Score"] >= 3:
            return "High Value At Risk"
        if row["R_Score"] <= 2 and row["F_Score"] <= 2:
            return "Lost Customers"
        return "Regular Customers"

    rfm["Segment"] = rfm.apply(assign_segment, axis=1)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(scaled)

    return rfm


df = create_dataset()
rfm = create_customer_analysis(df)

# ------------------------------------------------------------
# COMMON HELPERS
# ------------------------------------------------------------
def money(value):
    return f"${value:,.2f}"

def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_fig(fig, height=370):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(140,160,200,.12)", zeroline=False)
    return fig

def hero(title, subtitle, kicker="CUSTOMER INTELLIGENCE"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# SIDEBAR NAVIGATION + FILTERS
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛍️ ShopSense")
    st.caption("E-Commerce Intelligence")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Sales Analytics",
            "Customer Intelligence",
            "RFM Explorer",
            "K-Means Clustering",
            "Business Insights",
            "Dataset",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("### Global filters")
    chosen_countries = st.multiselect(
        "Countries",
        options=sorted(df["Country"].unique()),
        default=sorted(df["Country"].unique()),
    )
    chosen_categories = st.multiselect(
        "Categories",
        options=sorted(df["Category"].unique()),
        default=sorted(df["Category"].unique()),
    )
    st.caption("Demo data: 10,000 synthetic transactions from 2025.")
    if st.button("Regenerate demo data", use_container_width=True):
        create_dataset.clear()
        create_customer_analysis.clear()
        st.rerun()

filtered = df[
    df["Country"].isin(chosen_countries)
    & df["Category"].isin(chosen_categories)
].copy()

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
if page == "Overview":
    hero(
        "E-Commerce Intelligence Dashboard",
        "A live overview of sales performance, customer value, and purchasing behavior.",
    )
    revenue = filtered["Revenue"].sum()
    customers = filtered["Customer_ID"].nunique()
    transactions = len(filtered)
    units = int(filtered["Quantity"].sum())
    avg_order = filtered["Revenue"].mean() if transactions else 0

    cols = st.columns(5)
    with cols[0]:
        metric_card("TOTAL REVENUE", money(revenue), "Filtered sales")
    with cols[1]:
        metric_card("CUSTOMERS", f"{customers:,}", "Unique customers")
    with cols[2]:
        metric_card("TRANSACTIONS", f"{transactions:,}", "Purchase records")
    with cols[3]:
        metric_card("UNITS SOLD", f"{units:,}", "Total quantity")
    with cols[4]:
        metric_card("AVG. TRANSACTION", money(avg_order), "Average revenue per record")

    st.markdown("")
    left, right = st.columns([1.55, 1])
    with left:
        st.markdown('<div class="section-title">Monthly revenue trend</div>', unsafe_allow_html=True)
        monthly = filtered.groupby("Month", as_index=False)["Revenue"].sum()
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly["MonthName"] = monthly["Month"].apply(lambda x: month_names[int(x)-1])
        fig = px.line(monthly, x="MonthName", y="Revenue", markers=True)
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Revenue by category</div>', unsafe_allow_html=True)
        cat = filtered.groupby("Category", as_index=False)["Revenue"].sum()
        fig = px.pie(cat, names="Category", values="Revenue", hole=.62)
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(style_fig(fig, 390), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Top products by revenue</div>', unsafe_allow_html=True)
        top_products = filtered.groupby("Product", as_index=False)["Revenue"].sum().nlargest(7, "Revenue")
        fig = px.bar(top_products.sort_values("Revenue"), x="Revenue", y="Product", orientation="h")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Revenue by country</div>', unsafe_allow_html=True)
        country = filtered.groupby("Country", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
        fig = px.bar(country, x="Country", y="Revenue")
        st.plotly_chart(style_fig(fig, 350), use_container_width=True)

# ------------------------------------------------------------
# SALES ANALYTICS
# ------------------------------------------------------------
elif page == "Sales Analytics":
    hero("Sales Analytics", "Explore product performance, category mix, geography, and sales timing.", "SALES PERFORMANCE")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Product revenue ranking</div>', unsafe_allow_html=True)
        product = filtered.groupby("Product", as_index=False).agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"))
        fig = px.bar(product.sort_values("Revenue"), x="Revenue", y="Product", orientation="h", hover_data=["Units"])
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Units sold by product</div>', unsafe_allow_html=True)
        units = filtered.groupby("Product", as_index=False)["Quantity"].sum().sort_values("Quantity", ascending=False)
        fig = px.bar(units, x="Product", y="Quantity")
        st.plotly_chart(style_fig(fig, 440), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Category revenue share</div>', unsafe_allow_html=True)
        cat = filtered.groupby("Category", as_index=False)["Revenue"].sum()
        fig = px.pie(cat, names="Category", values="Revenue", hole=.45)
        st.plotly_chart(style_fig(fig, 370), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Revenue by weekday</div>', unsafe_allow_html=True)
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday = filtered.groupby("Weekday", as_index=False)["Revenue"].sum()
        weekday["Weekday"] = pd.Categorical(weekday["Weekday"], categories=weekday_order, ordered=True)
        weekday = weekday.sort_values("Weekday")
        fig = px.bar(weekday, x="Weekday", y="Revenue")
        st.plotly_chart(style_fig(fig, 370), use_container_width=True)

    st.markdown('<div class="section-title">Product performance table</div>', unsafe_allow_html=True)
    product_table = filtered.groupby(["Product", "Category"], as_index=False).agg(
        Revenue=("Revenue", "sum"),
        Units_Sold=("Quantity", "sum"),
        Transactions=("Product", "size"),
        Average_Unit_Price=("UnitPrice", "mean"),
    ).sort_values("Revenue", ascending=False)
    st.dataframe(product_table, use_container_width=True, hide_index=True)

    st.markdown("#### Explore a product")
    product_choice = st.selectbox("Choose product", ["All Products"] + sorted(filtered["Product"].unique()))
    temp = filtered if product_choice == "All Products" else filtered[filtered["Product"] == product_choice]
    monthly = temp.groupby("Month", as_index=False)["Revenue"].sum()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly["MonthName"] = monthly["Month"].apply(lambda x: month_names[int(x)-1])
    fig = px.line(monthly, x="MonthName", y="Revenue", markers=True, title=f"Monthly revenue — {product_choice}")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

# ------------------------------------------------------------
# CUSTOMER INTELLIGENCE
# ------------------------------------------------------------
elif page == "Customer Intelligence":
    hero("Customer Intelligence", "Understand customer value, repeat purchases, and behavioral patterns.", "CUSTOMER ANALYTICS")
    seg_counts = rfm["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
    seg_revenue = rfm.groupby("Segment", as_index=False)["Monetary"].sum().sort_values("Monetary", ascending=False)

    a, b, c = st.columns(3)
    with a:
        metric_card("CUSTOMER SEGMENTS", f"{rfm['Segment'].nunique()}", "Rule-based RFM groups")
    with b:
        metric_card("CHAMPIONS", f"{(rfm['Segment'] == 'Champions').sum():,}", "Recent, frequent, high spend")
    with c:
        metric_card("AT-RISK CUSTOMERS", f"{(rfm['Segment'] == 'At Risk').sum():,}", "May need re-engagement")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">Customer segment distribution</div>', unsafe_allow_html=True)
        fig = px.pie(seg_counts, names="Segment", values="Customers", hole=.45)
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    with right:
        st.markdown('<div class="section-title">Revenue contribution by segment</div>', unsafe_allow_html=True)
        fig = px.bar(seg_revenue, x="Segment", y="Monetary")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

    st.markdown('<div class="section-title">Customer purchase behavior</div>', unsafe_allow_html=True)
    fig = px.scatter(
        rfm, x="Frequency", y="Monetary", size="Monetary", color="Segment",
        hover_data=["Customer_ID", "Recency", "RFM_Score"],
        size_max=28,
    )
    st.plotly_chart(style_fig(fig, 470), use_container_width=True)

    st.markdown('<div class="section-title">Customer purchase frequency</div>', unsafe_allow_html=True)
    frequency = rfm.groupby("Frequency", as_index=False).size().rename(columns={"size": "Customers"})
    fig = px.bar(frequency, x="Frequency", y="Customers")
    st.plotly_chart(style_fig(fig, 330), use_container_width=True)

# ------------------------------------------------------------
# RFM EXPLORER
# ------------------------------------------------------------
elif page == "RFM Explorer":
    hero("RFM Customer Explorer", "Filter customer segments and find high-value, loyal, or at-risk customers.", "RFM ANALYSIS")
    segment_options = ["All Segments"] + sorted(rfm["Segment"].unique().tolist())
    selected_segment = st.selectbox("Customer segment", segment_options)
    sort_by = st.selectbox("Sort customers by", ["Monetary", "Frequency", "Recency"])
    ascending = sort_by == "Recency"
    customer_view = rfm.copy()
    if selected_segment != "All Segments":
        customer_view = customer_view[customer_view["Segment"] == selected_segment]

    a, b, c = st.columns(3)
    with a:
        metric_card("CUSTOMERS", f"{len(customer_view):,}", selected_segment)
    with b:
        metric_card("SEGMENT REVENUE", money(customer_view["Monetary"].sum()), "Total monetary value")
    with c:
        avg_recency = customer_view["Recency"].mean() if len(customer_view) else 0
        metric_card("AVG. RECENCY", f"{avg_recency:.1f} days", "Days since last purchase")

    customer_view = customer_view.sort_values(sort_by, ascending=ascending)
    st.markdown('<div class="section-title">Customer details</div>', unsafe_allow_html=True)
    st.dataframe(
        customer_view[["Customer_ID", "Recency", "Frequency", "Monetary", "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]]
        .style.format({"Monetary": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )
    csv = customer_view.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered customer data (CSV)", csv, "customer_rfm_analysis.csv", "text/csv")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">At-risk customers</div>', unsafe_allow_html=True)
        at_risk = rfm[rfm["Segment"].isin(["At Risk", "High Value At Risk"])].sort_values("Monetary", ascending=False).head(20)
        st.dataframe(at_risk[["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Score", "Segment"]], use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="section-title">Champion customers</div>', unsafe_allow_html=True)
        champions = rfm[rfm["Segment"] == "Champions"].sort_values("Monetary", ascending=False).head(20)
        st.dataframe(champions[["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Score"]], use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# K-MEANS CLUSTERING
# ------------------------------------------------------------
elif page == "K-Means Clustering":
    hero("K-Means Customer Segmentation", "Compare customers by recency, purchase frequency, and total spend using machine learning.", "MACHINE LEARNING")
    st.info("K-Means clusters are numbered 0–3. The numbers are cluster labels, not rankings. Use the summary table to interpret each group.")
    fig = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary", color="Cluster",
        hover_data=["Customer_ID", "Segment", "RFM_Score"],
        color_continuous_scale="Turbo",
    )
    fig.update_layout(scene=dict(
        xaxis_title="Recency (days)",
        yaxis_title="Purchase frequency",
        zaxis_title="Monetary value ($)",
    ))
    st.plotly_chart(style_fig(fig, 620), use_container_width=True)

    cluster_summary = rfm.groupby("Cluster").agg(
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum"),
        Customers=("Customer_ID", "count"),
    ).round(2).reset_index()
    st.markdown('<div class="section-title">Cluster summary</div>', unsafe_allow_html=True)
    st.dataframe(cluster_summary.style.format({
        "Average_Monetary": "${:,.2f}",
        "Total_Revenue": "${:,.2f}",
    }), use_container_width=True, hide_index=True)

    selected_cluster = st.selectbox("Inspect cluster", sorted(rfm["Cluster"].unique().tolist()))
    cluster_customers = rfm[rfm["Cluster"] == selected_cluster].sort_values("Monetary", ascending=False)
    st.markdown(f"#### Customers in cluster {selected_cluster}")
    st.dataframe(cluster_customers[["Customer_ID", "Recency", "Frequency", "Monetary", "Segment", "RFM_Score"]], use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# BUSINESS INSIGHTS
# ------------------------------------------------------------
elif page == "Business Insights":
    hero("Automatic Business Insights", "A quick summary of strongest performers and customer groups in the generated dataset.", "ACTIONABLE INSIGHTS")
    if filtered.empty:
        st.warning("No records match the selected global filters. Select at least one country and category.")
    else:
        best_product = filtered.groupby("Product")["Revenue"].sum().idxmax()
        best_country = filtered.groupby("Country")["Revenue"].sum().idxmax()
        best_category = filtered.groupby("Category")["Revenue"].sum().idxmax()
        largest_segment = rfm["Segment"].value_counts().idxmax()
        best_product_revenue = filtered.groupby("Product")["Revenue"].sum().max()
        best_country_revenue = filtered.groupby("Country")["Revenue"].sum().max()
        best_category_revenue = filtered.groupby("Category")["Revenue"].sum().max()

        cols = st.columns(3)
        with cols[0]:
            metric_card("TOP PRODUCT", best_product, money(best_product_revenue) + " revenue")
        with cols[1]:
            metric_card("TOP COUNTRY", best_country, money(best_country_revenue) + " revenue")
        with cols[2]:
            metric_card("TOP CATEGORY", best_category, money(best_category_revenue) + " revenue")

        st.markdown("")
        st.markdown(f"**Largest customer segment:** {largest_segment} ({(rfm['Segment'] == largest_segment).sum():,} customers).")
        at_risk_count = int(rfm["Segment"].isin(["At Risk", "High Value At Risk"]).sum())
        champion_count = int((rfm["Segment"] == "Champions").sum())
        st.markdown(f"- **Retention opportunity:** {at_risk_count:,} customers are labeled At Risk or High Value At Risk by the rule-based RFM model.")
        st.markdown(f"- **Loyalty opportunity:** {champion_count:,} customers are labeled Champions; consider loyalty rewards or early access.")
        st.markdown("- **Sales opportunity:** Review the product and country charts to see where revenue is concentrated.")
        st.caption("These are descriptive insights from synthetic demo data, not predictions about real customers.")

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
elif page == "Dataset":
    hero("Transaction Dataset", "Review the generated transaction records and download a copy for further analysis.", "DATA EXPLORER")
    st.write(f"**Rows:** {len(filtered):,}  |  **Columns:** {filtered.shape[1]}")
    search_product = st.selectbox("Filter by product", ["All Products"] + sorted(filtered["Product"].unique()))
    data_view = filtered if search_product == "All Products" else filtered[filtered["Product"] == search_product]
    st.dataframe(data_view, use_container_width=True, hide_index=True)
    csv = data_view.to_csv(index=False).encode("utf-8")
    st.download_button("Download transactions (CSV)", csv, "ecommerce_transactions.csv", "text/csv")

st.markdown("---")
st.markdown(
    '<div class="small-muted">ShopSense · E-Commerce Customer Intelligence · Built with Streamlit, Pandas, Scikit-learn and Plotly</div>',
    unsafe_allow_html=True,
)


