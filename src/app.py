import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

# Page Configuration
st.set_page_config(
    page_title="Developer Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a page",
    ("Predict", "Explore"),
    help="Select which page to view"
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About This App**

This application uses machine learning to predict developer salaries based on the Stack Overflow 2025 Developer Survey.

- **Predict**: Get salary predictions for your profile
- **Explore**: Discover insights from 49,000+ developer responses
""")


# Route to appropriate page
if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
