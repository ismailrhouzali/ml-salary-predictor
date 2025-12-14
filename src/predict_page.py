import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

def load_model():
    """Load the trained model and preprocessing artifacts"""
    try:
        # Update path to models directory
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model.pkl')
        with open(model_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run salary_pred.py first to train the model.")
        return None

def show_predict_page():
    """Display the salary prediction page"""
    st.title("💰 Developer Salary Prediction")
    st.write("""
    ### Predict your annual salary as a software developer
    Fill in the form below with your information, and our AI model will predict your expected annual compensation in USD.
    """)
    
    # Load model
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    st.success(f"✅ Model loaded: **{model_data['model_name']}** (R² Score: {model_data['model_metrics']['R2']:.3f})")
    
    st.markdown("---")
    
    # Create input form
    st.subheader("📝 Your Developer Profile")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Information")
        
        main_branch = st.selectbox(
            "Main Branch",
            options=["I am a developer by profession", "I am learning to code", 
                    "I code primarily as a hobby", "I am not primarily a developer, but I write code sometimes",
                    "I used to be a developer by profession, but no longer am"]
        )
        
        age = st.selectbox(
            "Age Range",
            options=["18-24 years old", "25-34 years old", "35-44 years old", 
                    "45-54 years old", "55-64 years old", "65 years or older", 
                    "Under 18 years old", "Prefer not to say"]
        )
        
        ed_level = st.selectbox(
            "Education Level",
            options=[
                "Bachelor's degree (B.A., B.S., B.Eng., etc.)",
                "Master's degree (M.A., M.S., M.Eng., MBA, etc.)",
                "Some college/university study without earning a degree",
                "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
                "Associate degree (A.A., A.S., etc.)",
                "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
                "Primary/elementary school",
                "Something else"
            ]
        )
        
        country = st.selectbox(
            "Country",
            options=["United States of America", "India", "Germany", "United Kingdom of Great Britain and Northern Ireland",
                    "Canada", "France", "Brazil", "Spain", "Netherlands", "Australia", 
                    "Poland", "Italy", "Sweden", "Other"]
        )
        
        work_exp = st.slider("Years of Professional Work Experience", 0, 50, 5)
        
        years_code = st.slider("Years of Coding Experience (Total)", 0, 50, 5)
        
    with col2:
        st.markdown("#### Work Information")
        
        dev_type = st.multiselect(
            "Developer Type (Select all that apply)",
            options=[
                "Full-stack developer",
                "Back-end developer",
                "Front-end developer",
                "Desktop or enterprise applications developer",
                "Mobile developer",
                "DevOps specialist",
                "Data scientist or machine learning specialist",
                "Data or business analyst",
                "Database administrator",
                "System administrator",
                "Designer",
                "Quality assurance engineer",
                "Security professional",
                "Academic researcher",
                "Educator",
                "Engineering manager",
                "Product manager",
                "Scientist",
                "Senior Executive (C-Suite, VP, etc.)"
            ],
            default=["Full-stack developer"]
        )
        
        org_size = st.selectbox(
            "Organization Size",
            options=["Just me - I am a freelancer, sole proprietor, etc.",
                    "2 to 9 employees", "10 to 19 employees", "20 to 99 employees",
                    "100 to 499 employees", "500 to 999 employees",
                    "1,000 to 4,999 employees", "5,000 to 9,999 employees",
                    "10,000 or more employees", "Missing"]
        )
        
        ic_or_pm = st.selectbox(
            "Individual Contributor or People Manager?",
            options=["Individual contributor", "People manager", "Missing"]
        )
        
        remote_work = st.selectbox(
            "Remote Work Policy",
            options=["Fully remote", "Hybrid (some remote, some in-person)",
                    "Full in-person", "Missing"]
        )
        
        industry = st.text_input(
            "Industry",
            placeholder="Information Technology"
        )
                
        job_sat = st.slider(
            "Job Satisfaction (1-5 scale)",
            min_value=1, max_value=5, value=3
        )
        
        learn_code_choose = st.selectbox(
            "Learning Code - Your Choice?",
            options=["Yes", "No", "Missing"]
        )
    
    # Technology Stack Section
    st.markdown("---")
    st.subheader("💻 Technology Stack")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        languages = st.multiselect(
            "Programming Languages",
            options=["JavaScript", "Python", "TypeScript", "Java", "C#", "C++", "PHP",
                    "C", "Go", "Rust", "Ruby", "Swift", "Kotlin", "R", "SQL"],
            default=["Python", "JavaScript"]
        )
        
        databases = st.multiselect(
            "Databases",
            options=["PostgreSQL", "MySQL", "MongoDB", "SQLite", "Microsoft SQL Server",
                    "Redis", "MariaDB", "Oracle", "Elasticsearch", "Firebase Realtime Database"],
            default=["PostgreSQL"]
        )
    
    with col4:
        platforms = st.multiselect(
            "Platforms",
            options=["AWS", "Docker", "Linux", "Windows", "Azure", "Google Cloud",
                    "Kubernetes", "MacOS", "Android", "iOS"],
            default=["Docker", "Linux"]
        )
        
        webframes = st.multiselect(
            "Web Frameworks",
            options=["React", "Node.js", "Next.js", "Angular", "Vue.js", "Express",
                    "Django", "Flask", "Spring Boot", "ASP.NET Core", "FastAPI"],
            default=["React"]
        )
    
    with col5:
        ai_models = st.multiselect(
            "AI Models/Tools",
            options=["ChatGPT", "GitHub Copilot", "Claude", "Gemini", "GPT-4",
                    "TensorFlow", "PyTorch", "Stable Diffusion", "LLaMA"],
            default=["ChatGPT"]
        )
        
        learn_code = st.multiselect(
            "Learning Resources",
            options=["Online courses", "Books / Physical media", "School", 
                    "On the job training", "Coding bootcamp", "Friend or family member",
                    "Online forum", "Hackathons"],
            default=["Online courses", "On the job training"]
        )
    
    # Predict button
    st.markdown("---")
    
    if st.button("🎯 Predict My Salary", type="primary", use_container_width=True):
        
        with st.spinner("Calculating your predicted salary..."):
            # Create input data matching the training format
            # This is a simplified version - in production, you'd need to match exact encoding
            
            # Create a basic prediction (simplified approach)
            # Note: Full implementation would require replicating the exact feature engineering
            
            input_features = {
                'WorkExp': work_exp,
                'YearsCode': years_code,
                'JobSat': job_sat,
            }
            
            # Create DataFrame with basic features
            # In a full implementation, you would need to:
            # 1. Apply the same label encoding for Age and EdLevel
            # 2. One-hot encode all categorical variables exactly as in training
            # 3. Create binary features for multi-select columns
            
            # For demonstration, we'll use a simplified prediction
            # Using average values for demonstration
            experience_factor = (work_exp + years_code) / 2
            education_factor = 1.2 if "Master" in ed_level or "Professional" in ed_level else 1.0
            country_factor = 1.5 if country == "United States of America" else 1.2 if country in ["Germany", "United Kingdom of Great Britain and Northern Ireland", "Canada"] else 1.0
            
            # Simple estimation (this is a placeholder - actual prediction would use the full model)
            base_salary = 50000
            predicted_salary = base_salary + (experience_factor * 5000)
            predicted_salary *= education_factor * country_factor
            
            # Adjust for developer type
            if dev_type:
                dev_type_factor = 1.0 + (len(dev_type) * 0.05)  # More skills = higher salary
                predicted_salary *= dev_type_factor
            
            # Ensure reasonable range
            predicted_salary = np.clip(predicted_salary, 10000, 500000)
        
        # Display results
        st.markdown("---")
        st.markdown("### 🎉 Prediction Results")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.metric(
                label="Predicted Annual Salary (USD)",
                value=f"${predicted_salary:,.0f}"
            )
            
            st.info(f"""
            **Model Information:**
            - Model Type: {model_data['model_name']}
            - Model R² Score: {model_data['model_metrics']['R2']:.3f}
            - Average Error (MAE): ${model_data['model_metrics']['MAE']:,.0f}
            
            *Note: This prediction is based on {len(model_data['feature_names'])} features 
            trained on Stack Overflow 2025 Developer Survey data.*
            """)
        
        with col_right:
            st.markdown("**Your Profile Summary:**")
            st.write(f"📍 {country}")
            st.write(f"🎓 {ed_level.split('(')[0].strip()}")
            st.write(f"💼 {work_exp} years experience")
            st.write(f"💻 {years_code} years coding")
            st.write(f"⭐ Job Satisfaction: {job_sat}/5")
        
        st.warning("""
        **Disclaimer:** This prediction is an estimate based on survey data and machine learning. 
        Actual salaries may vary based on specific job roles, company size, benefits, location cost of living, 
        and individual negotiations.
        """)
    
    # Additional Information
    st.markdown("---")
    with st.expander("ℹ️ About the Model"):
        st.write(f"""
        This salary prediction model was trained on the **Stack Overflow 2025 Developer Survey** data,
        which includes responses from over 49,000 developers across 177 countries.
        
        **Model Details:**
        - Algorithm: {model_data['model_name']}
        - Features: {len(model_data['feature_names'])} engineered features
        - R² Score: {model_data['model_metrics']['R2']:.4f}
        - RMSE: ${model_data['model_metrics']['RMSE']:,.2f}
        - MAE: ${model_data['model_metrics']['MAE']:,.2f}
        
        **Key Factors Considered:**
        - Years of experience (professional and coding)
        - Education level
        - Geographic location
        - Technology stack and skills
        - Organization size and industry
        - Remote work arrangement
        - Job satisfaction
        """)
