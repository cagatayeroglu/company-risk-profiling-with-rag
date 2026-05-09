import os
import sys
import pickle
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from src.comparator import RiskComparator
from config import COMPANIES, RISK_CATEGORIES, EMBEDDINGS_DIR

# ============================================================
# Page Configuration & Styling
# ============================================================
st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .evidence-box {
        background-color: #f0f2f6;
        color: #1a1c24; /* Fix for dark mode white text */
        border-left: 4px solid #4e4376;
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9em;
        white-space: pre-wrap;
    }
    .metric-card {
        background-color: #f4f4f6;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Data Loading
# ============================================================
@st.cache_data
def load_data():
    comp = RiskComparator()
    return comp

@st.cache_data
def load_chunk_metadata():
    """Load metadata to map Chunk IDs back to full text."""
    meta_path = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            # Create a lookup by chunk_id string
            return {meta["chunk_id"]: meta["text"] for meta in metadata.values()}
    return {}

comparator = load_data()
chunk_lookup = load_chunk_metadata()
available_companies = comparator.get_available_companies()

if not available_companies:
    st.error("No risk profiles found. Please run the LLM extraction or generate mock profiles first.")
    st.stop()

# ============================================================
# Sidebar
# ============================================================
st.sidebar.title("Navigation")
st.sidebar.markdown("Explore automated risk profiles extracted from Form 10-K filings using Multi-Document RAG.")
st.sidebar.markdown("---")

st.sidebar.markdown("**Available Companies:**")
for ticker in available_companies:
    name = COMPANIES.get(ticker, ticker)
    st.sidebar.markdown(f"- **{ticker}**: {name}")

# ============================================================
# Main Header
# ============================================================
st.markdown('<p class="main-header">Automated Risk Profiling Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Qwen2.5-3B, FAISS, and BAAI/bge-small-en-v1.5</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Risk Heatmap", 
    "🏆 Top Risks by Company", 
    "🔍 Evidence Explorer", 
    "⚖️ Company Comparison"
])

# ============================================================
# Tab 1: Risk Heatmap
# ============================================================
with tab1:
    st.markdown("### Industry Risk Overview")
    st.markdown("A macro view of risk severity across all analyzed companies and categories.")
    
    df_scores = comparator.get_risk_heatmap_data()
    df_labels = comparator.get_severity_labels_matrix()
    
    if not df_scores.empty:
        # Create a custom heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=df_scores.values,
            x=df_scores.columns,
            y=df_scores.index,
            text=df_labels.values,
            texttemplate="%{text}",
            colorscale=[[0, "#f8f9fa"], [0.33, "#e8f5e9"], [0.66, "#fff3e0"], [1.0, "#ffebee"]],
            showscale=False,
            hoverinfo="x+y+text"
        ))
        
        fig.update_layout(
            height=500,
            xaxis=dict(tickangle=-45),
            margin=dict(t=30, l=100, r=20, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for heatmap.")

# ============================================================
# Tab 2: Top Risks by Company
# ============================================================
with tab2:
    st.markdown("### Company Deep Dive")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Select Company", available_companies, key="top_risks_company")
        
    profile = comparator.get_company_profile(selected_ticker)
    
    if profile:
        with col2:
            st.markdown(f"#### {profile['company_name']} ({selected_ticker})")
            st.markdown(f"**Total Risks Identified:** {profile['risks_found']} out of {profile['total_categories']} categories.")
            
        top_risks = comparator.get_top_risks_for_company(selected_ticker, top_n=5)
        
        for i, risk in enumerate(top_risks):
            sev_class = f"risk-{risk['severity'].lower()}"
            with st.expander(f"{i+1}. {risk['risk_category']} - {risk['severity'].upper()}", expanded=(i==0)):
                st.markdown(f"**Severity:** <span class='{sev_class}'>{risk['severity'].upper()}</span> | **Confidence:** {risk['confidence']:.2f}", unsafe_allow_html=True)
                st.markdown(f"**LLM Assessment:** {risk['explanation']}")
                
                if risk.get("evidence_snippets"):
                    st.markdown("**Key Evidence:**")
                    for snippet in risk["evidence_snippets"]:
                        # Some LLMs return the Chunk ID instead of the text, e.g. "[Chunk 1] (ID: AAPL_2025_item1a_0031)"
                        chunk_id_match = re.search(r"ID:\s*([A-Za-z0-9_]+)", snippet)
                        if chunk_id_match:
                            chunk_id = chunk_id_match.group(1)
                            full_text = chunk_lookup.get(chunk_id, snippet)
                            with st.expander(f"📄 View Original 10-K Chunk: {chunk_id}", expanded=False):
                                st.markdown(f"<div class='evidence-box'>{full_text}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='evidence-box'>\"{snippet}\"</div>", unsafe_allow_html=True)

# ============================================================
# Tab 3: Evidence Explorer
# ============================================================
with tab3:
    st.markdown("### Source Traceability")
    st.markdown("Investigate the exact 10-K evidence snippets the LLM used to make its assessment.")
    
    col1, col2 = st.columns(2)
    with col1:
        ev_company = st.selectbox("Select Company", available_companies, key="ev_company")
    with col2:
        ev_category = st.selectbox(
            "Select Risk Category", 
            [cat["name"] for cat in RISK_CATEGORIES], 
            key="ev_category"
        )
        
    profile = comparator.get_company_profile(ev_company)
    if profile:
        risk_data = next((r for r in profile["risk_assessments"] if r["risk_category"] == ev_category), None)
        
        if risk_data:
            if not risk_data["is_present"]:
                st.info(f"The LLM determined that **{ev_category}** is not significantly mentioned for {ev_company}.")
            else:
                st.markdown(f"#### Assessment: <span class='risk-{risk_data['severity'].lower()}'>{risk_data['severity'].upper()}</span> Risk", unsafe_allow_html=True)
                st.write(risk_data["explanation"])
                
                st.markdown("#### Retrieved Source Evidence (Grounding)")
                for idx, snippet in enumerate(risk_data.get("evidence_snippets", [])):
                    chunk_id_match = re.search(r"ID:\s*([A-Za-z0-9_]+)", snippet)
                    if chunk_id_match:
                        chunk_id = chunk_id_match.group(1)
                        full_text = chunk_lookup.get(chunk_id, snippet)
                        st.markdown(f"**Snippet {idx+1} (Original 10-K Context: {chunk_id})**")
                        st.markdown(f"<div class='evidence-box'>{full_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Snippet {idx+1}**")
                        st.markdown(f"<div class='evidence-box'>\"{snippet}\"</div>", unsafe_allow_html=True)
        else:
            st.warning("No data found for this combination.")

# ============================================================
# Tab 4: Company Comparison
# ============================================================
with tab4:
    st.markdown("### Side-by-Side Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        comp1 = st.selectbox("Company 1", available_companies, index=0)
    with col2:
        comp2 = st.selectbox("Company 2", available_companies, index=1 if len(available_companies) > 1 else 0)
        
    if comp1 and comp2:
        df_comp = comparator.compare_two_companies(comp1, comp2)
        
        # Display nicely
        for _, row in df_comp.iterrows():
            st.markdown(f"#### {row['Risk Category']}")
            
            c1, c2 = st.columns(2)
            with c1:
                sev1 = row[f"{comp1} Severity"]
                sev_class1 = f"risk-{sev1.lower()}" if sev1 != "None" else ""
                st.markdown(f"**{comp1}**: <span class='{sev_class1}'>{sev1}</span>", unsafe_allow_html=True)
                if sev1 != "None":
                    st.write(row[f"{comp1} Explanation"])
                    
            with c2:
                sev2 = row[f"{comp2} Severity"]
                sev_class2 = f"risk-{sev2.lower()}" if sev2 != "None" else ""
                st.markdown(f"**{comp2}**: <span class='{sev_class2}'>{sev2}</span>", unsafe_allow_html=True)
                if sev2 != "None":
                    st.write(row[f"{comp2} Explanation"])
                    
            st.markdown("---")
