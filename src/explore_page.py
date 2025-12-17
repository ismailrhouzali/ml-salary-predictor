import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os

# Set page configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        # Try to load from local file first (for local development)
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stack-overflow-developer-survey-2025', 'survey_results_public.csv')
        
        if os.path.exists(data_path):
            dfc = pd.read_csv(data_path, low_memory=False)
        else:
            # Fallback: Download from GitHub LFS URL (for Streamlit Cloud)
            st.info("📥 Downloading dataset from GitHub (this may take a moment)...")
            
            # GitHub raw URL for the CSV file (Git LFS pointer will redirect to actual file)
            csv_url = "https://media.githubusercontent.com/media/ismailrhouzali/ml-salary-predictor/main/data/stack-overflow-developer-survey-2025/survey_results_public.csv"
            
            try:
                dfc = pd.read_csv(csv_url, low_memory=False)
                st.success("✅ Dataset loaded successfully from GitHub!")
            except Exception as e:
                st.error(f"❌ Failed to download dataset: {str(e)}")
                st.info("""
                **Note:** The Explore page requires the full dataset which is too large for standard GitHub.
                
                For local development:
                1. Download the Stack Overflow 2025 survey data
                2. Place it in `data/stack-overflow-developer-survey-2025/`
                3. Run the app locally with `python run.py`
                """)
                return None
        
        # Rename target column
        dfc = dfc.rename(columns={'ConvertedCompYearly': 'AnnualCompUSD'})
        
        # Initial filtering
        dfc = dfc[dfc["AnnualCompUSD"].notnull()]
        dfc = dfc[dfc["Employment"] == "Employed"]
        
        # Handle missing values
        # WorkExp: NaN means no professional work experience, fill with 0
        dfc['WorkExp'] = dfc['WorkExp'].fillna(0)
        dfc['YearsCode'] = dfc['YearsCode'].fillna(dfc['YearsCode'].median())
        
        # Remove outliers
        dfc = dfc[(dfc['AnnualCompUSD'] >= 10000) & (dfc['AnnualCompUSD'] <= 500000)]
        
        return dfc
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def analyze_multi_select_column(series, top_n=15):
    """Analyze semicolon-separated multi-select columns"""
    series_cleaned = series.dropna().astype(str)
    list_of_lists = series_cleaned.apply(
        lambda x: [item.strip() for item in x.split(';')]
    ).tolist()
    all_items = [item for sublist in list_of_lists for item in sublist]
    frequency_count = Counter(all_items)
    df_freq = pd.DataFrame(
        frequency_count.items(),
        columns=['Element', 'Frequency']
    ).sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return df_freq.head(top_n)

def show_explore_page():
    """Display comprehensive data exploration page"""
    
    st.title("📊 Developer Salary Data Explorer")
    st.write("""
    Explore comprehensive insights from the **Stack Overflow 2025 Developer Survey** with 
    interactive visualizations covering salary trends, demographics, technology stacks, and more.
    """)
    
    # Load data
    with st.spinner("Loading dataset..."):
        dfc = load_data()
    
    if dfc is None:
        st.stop()
    
    st.success(f"✅ Loaded {len(dfc):,} developer responses")
    
    # ========================================================================
    # SECTION 1: DATASET OVERVIEW
    # ========================================================================
    
    st.markdown("---")
    st.header("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Responses", f"{len(dfc):,}")
    with col2:
        st.metric("Features", f"{dfc.shape[1]}")
    with col3:
        st.metric("Countries", f"{dfc['Country'].nunique()}")
    with col4:
        st.metric("Avg Salary", f"${dfc['AnnualCompUSD'].mean():,.0f}")
    
    # Data Viewer
    with st.expander("🔍 View Raw Data (First 100 Rows)"):
        st.dataframe(dfc.head(100), use_container_width=True)
    
    # Missing Values
    with st.expander("📊 Missing Values Analysis"):
        missing_df = pd.DataFrame({
            'Column': dfc.columns,
            'Missing Count': dfc.isnull().sum().values,
            'Missing %': (dfc.isnull().sum().values / len(dfc) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            fig = px.bar(missing_df, x='Missing %', y='Column', orientation='h',
                        title='Missing Values by Column',
                        color='Missing %',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the filtered dataset!")
    
    # ========================================================================
    # SECTION 2: SALARY ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("💰 Salary Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary Distribution
        fig1 = px.histogram(dfc, x='AnnualCompUSD', nbins=50,
                           title='Annual Salary Distribution',
                           labels={'AnnualCompUSD': 'Annual Compensation (USD)'},
                           color_discrete_sequence=['#1f77b4'])
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Salary Box Plot
        fig2 = px.box(dfc, y='AnnualCompUSD',
                     title='Salary Distribution (Box Plot)',
                     labels={'AnnualCompUSD': 'Annual Compensation (USD)'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Salary Statistics
    st.subheader("📈 Salary Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    with stat_col1:
        st.metric("Minimum", f"${dfc['AnnualCompUSD'].min():,.0f}")
    with stat_col2:
        st.metric("25th Percentile", f"${dfc['AnnualCompUSD'].quantile(0.25):,.0f}")
    with stat_col3:
        st.metric("Median", f"${dfc['AnnualCompUSD'].median():,.0f}")
    with stat_col4:
        st.metric("75th Percentile", f"${dfc['AnnualCompUSD'].quantile(0.75):,.0f}")
    with stat_col5:
        st.metric("Maximum", f"${dfc['AnnualCompUSD'].max():,.0f}")
    
    # ========================================================================
    # SECTION 3: GEOGRAPHIC ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("🌍 Geographic Distribution")
    
    # Country analysis
    country_counts = dfc['Country'].value_counts().head(15)
    country_salary = dfc.groupby('Country')['AnnualCompUSD'].agg(['mean', 'median', 'count']).reset_index()
    country_salary = country_salary[country_salary['count'] >= 50].sort_values('mean', ascending=False).head(15)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(country_counts, 
                    title='Top 15 Countries by Developer Count',
                    labels={'value': 'Number of Developers', 'Country': 'Country'},
                    color=country_counts.values,
                    color_continuous_scale='Blues')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(country_salary, x='Country', y='mean',
                    title='Average Salary by Country (Top 15)',
                    labels={'mean': 'Average Salary (USD)', 'Country': 'Country'},
                    color='mean',
                    color_continuous_scale='Greens')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SECTION 4: DEMOGRAPHIC ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("👥 Demographics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Age", "Education", "Experience", "Organization"])
    
    with tab1:
        # Age Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            age_counts = dfc['Age'].value_counts()
            fig = px.pie(values=age_counts.values, names=age_counts.index,
                        title='Age Distribution',
                        color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            age_salary = dfc.groupby('Age')['AnnualCompUSD'].mean().sort_values(ascending=False)
            fig = px.bar(age_salary, 
                        title='Average Salary by Age Group',
                        labels={'value': 'Average Salary (USD)', 'Age': 'Age Group'},
                        color=age_salary.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Education Level
        col1, col2 = st.columns(2)
        
        with col1:
            ed_counts = dfc['EdLevel'].value_counts().head(10)
            fig = px.bar(ed_counts,
                        title='Education Level Distribution (Top 10)',
                        labels={'value': 'Count', 'EdLevel': 'Education Level'},
                        color=ed_counts.values,
                        color_continuous_scale='Purples')
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            ed_salary = dfc.groupby('EdLevel')['AnnualCompUSD'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(ed_salary,
                        title='Average Salary by Education (Top 10)',
                        labels={'value': 'Average Salary (USD)', 'EdLevel': 'Education Level'},
                        color=ed_salary.values,
                        color_continuous_scale='Oranges')
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Experience Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Work Experience Distribution
            fig = px.histogram(dfc, x='WorkExp', nbins=30,
                             title='Work Experience Distribution',
                             labels={'WorkExp': 'Years of Professional Experience'},
                             color_discrete_sequence=['#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Years of Coding
            fig = px.histogram(dfc, x='YearsCode', nbins=30,
                             title='Years of Coding Distribution',
                             labels={'YearsCode': 'Years of Coding Experience'},
                             color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Experience vs Salary Scatter
        sample_data = dfc.sample(min(1000, len(dfc)))  # Sample for performance
        fig = px.scatter(sample_data, x='WorkExp', y='AnnualCompUSD',
                        title='Work Experience vs Salary (Sample of 1000)',
                        labels={'WorkExp': 'Years of Experience', 'AnnualCompUSD': 'Annual Salary (USD)'},
                        opacity=0.6,
                        color='YearsCode',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Organization Size
        col1, col2 = st.columns(2)
        
        with col1:
            org_counts = dfc['OrgSize'].value_counts()
            fig = px.pie(values=org_counts.values, names=org_counts.index,
                        title='Organization Size Distribution',
                        color_discrete_sequence=px.colors.sequential.Teal)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            org_salary = dfc.groupby('OrgSize')['AnnualCompUSD'].mean().sort_values(ascending=False)
            fig = px.bar(org_salary,
                        title='Average Salary by Organization Size',
                        labels={'value': 'Average Salary (USD)', 'OrgSize': 'Organization Size'},
                        color=org_salary.values,
                        color_continuous_scale='Teal')
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SECTION 5: WORK ENVIRONMENT
    # ========================================================================
    
    st.markdown("---")
    st.header("🏢 Work Environment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Remote Work
        remote_counts = dfc['RemoteWork'].value_counts()
        fig = px.pie(values=remote_counts.values, names=remote_counts.index,
                    title='Remote Work Distribution',
                    color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IC or PM
        ic_counts = dfc['ICorPM'].value_counts()
        fig = px.pie(values=ic_counts.values, names=ic_counts.index,
                    title='Individual Contributor vs Manager',
                    color_discrete_sequence=px.colors.sequential.Sunset)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Job Satisfaction
        jobsat_counts = dfc['JobSat'].value_counts().sort_index()
        fig = px.bar(jobsat_counts,
                    title='Job Satisfaction Distribution',
                    labels={'value': 'Count', 'JobSat': 'Job Satisfaction (1-5)'},
                    color=jobsat_counts.values,
                    color_continuous_scale='RdYlGn')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Remote Work vs Salary
    remote_salary = dfc.groupby('RemoteWork')['AnnualCompUSD'].mean().sort_values(ascending=False)
    fig = px.bar(remote_salary,
                title='Average Salary by Remote Work Policy',
                labels={'value': 'Average Salary (USD)', 'RemoteWork': 'Remote Work Policy'},
                color=remote_salary.values,
                color_continuous_scale='Blues')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SECTION 6: TECHNOLOGY STACK ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("💻 Technology Stack Analysis")
    
    tech_tab1, tech_tab2, tech_tab3, tech_tab4, tech_tab5 = st.tabs([
        "Programming Languages", "Databases", "Platforms", "Web Frameworks", "AI Models"
    ])
    
    with tech_tab1:
        st.subheader("Most Popular Programming Languages")
        lang_freq = analyze_multi_select_column(dfc['LanguageHaveWorkedWith'], top_n=20)
        
        fig = px.bar(lang_freq, x='Frequency', y='Element', orientation='h',
                    title='Top 20 Programming Languages',
                    labels={'Frequency': 'Number of Developers', 'Element': 'Language'},
                    color='Frequency',
                    color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(lang_freq, use_container_width=True)
    
    with tech_tab2:
        st.subheader("Most Popular Databases")
        db_freq = analyze_multi_select_column(dfc['DatabaseHaveWorkedWith'], top_n=20)
        
        fig = px.bar(db_freq, x='Frequency', y='Element', orientation='h',
                    title='Top 20 Databases',
                    labels={'Frequency': 'Number of Developers', 'Element': 'Database'},
                    color='Frequency',
                    color_continuous_scale='Greens')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(db_freq, use_container_width=True)
    
    with tech_tab3:
        st.subheader("Most Popular Platforms")
        platform_freq = analyze_multi_select_column(dfc['PlatformHaveWorkedWith'], top_n=20)
        
        fig = px.bar(platform_freq, x='Frequency', y='Element', orientation='h',
                    title='Top 20 Platforms',
                    labels={'Frequency': 'Number of Developers', 'Element': 'Platform'},
                    color='Frequency',
                    color_continuous_scale='Oranges')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(platform_freq, use_container_width=True)
    
    with tech_tab4:
        st.subheader("Most Popular Web Frameworks")
        webframe_freq = analyze_multi_select_column(dfc['WebframeHaveWorkedWith'], top_n=20)
        
        fig = px.bar(webframe_freq, x='Frequency', y='Element', orientation='h',
                    title='Top 20 Web Frameworks',
                    labels={'Frequency': 'Number of Developers', 'Element': 'Framework'},
                    color='Frequency',
                    color_continuous_scale='Purples')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(webframe_freq, use_container_width=True)
    
    with tech_tab5:
        st.subheader("Most Popular AI Models/Tools")
        ai_freq = analyze_multi_select_column(dfc['AIModelsHaveWorkedWith'], top_n=20)
        
        fig = px.bar(ai_freq, x='Frequency', y='Element', orientation='h',
                    title='Top 20 AI Models/Tools',
                    labels={'Frequency': 'Number of Developers', 'Element': 'AI Model/Tool'},
                    color='Frequency',
                    color_continuous_scale='Reds')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(ai_freq, use_container_width=True)
    
    # ========================================================================
    # SECTION 7: LEARNING & DEVELOPMENT
    # ========================================================================
    
    st.markdown("---")
    st.header("📚 Learning & Development")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Learning Choice
        learn_choice_counts = dfc['LearnCodeChoose'].value_counts()
        fig = px.pie(values=learn_choice_counts.values, names=learn_choice_counts.index,
                    title='Learning to Code: Was it Your Choice?',
                    color_discrete_sequence=px.colors.sequential.ice)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Learning Methods
        learn_freq = analyze_multi_select_column(dfc['LearnCode'], top_n=15)
        fig = px.bar(learn_freq, x='Frequency', y='Element', orientation='h',
                    title='Most Common Learning Methods',
                    labels={'Frequency': 'Number of Developers', 'Element': 'Learning Method'},
                    color='Frequency',
                    color_continuous_scale='YlOrRd')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # SECTION 8: CORRELATION ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("🔗 Correlation Analysis")
    
    # Select numerical columns (excluding CompTotal as it's redundant with AnnualCompUSD)
    numerical_cols = ['WorkExp', 'YearsCode', 'JobSat', 'AnnualCompUSD']
    correlation_data = dfc[numerical_cols].corr()
    
    fig = px.imshow(correlation_data,
                    labels=dict(color="Correlation"),
                    x=numerical_cols,
                    y=numerical_cols,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title='Correlation Heatmap - Numerical Features')
    fig.update_xaxes(side="bottom")
    st.plotly_chart(fig, use_container_width=True)
    
    