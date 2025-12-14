# Developer Salary Prediction - Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Details](#model-details)
5. [Feature Engineering](#feature-engineering)
6. [API Reference](#api-reference)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)

---

## 1. Project Overview

### Purpose

Predict software developer salaries using machine learning based on Stack Overflow 2025 Developer Survey data.

### Key Metrics

- **Accuracy**: 88.49% (R² score)
- **Average Error**: $9,726 (MAE)
- **Dataset Size**: 17,679 employed developers
- **Features**: 267 engineered features
- **Countries**: 177

### Technology Stack

- **ML Framework**: XGBoost, scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Seaborn, Matplotlib

---

## 2. Architecture

### Project Structure

```
Salary_Predict/
├── src/                    # Source code
│   ├── app.py             # Main Streamlit application
│   ├── predict_page.py    # Prediction interface
│   ├── explore_page.py    # Data exploration dashboard
│   └── salary_pred.py     # ML training pipeline
├── models/                 # Trained models
│   └── saved_model.pkl    # Serialized XGBoost model
├── data/                   # Dataset storage
│   └── stack-overflow-developer-survey-2025/
│       └── survey_results_public.csv
├── notebooks/              # Jupyter notebooks
│   └── Draft.ipynb        # EDA and experimentation
├── docs/                   # Documentation
├── screenshots/            # Application screenshots
└── run.py                  # Application launcher
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)               │
├─────────────────────────────────────────────────────────────┤
│  Predict Page          │         Explore Page                │
│  - Input Form          │         - 40+ Visualizations        │
│  - Predictions         │         - Interactive Filters       │
└──────────┬─────────────┴─────────────────┬──────────────────┘
           │                               │
           ▼                               ▼
    ┌──────────────┐              ┌──────────────┐
    │ Trained Model│              │  Raw Dataset │
    │  (XGBoost)   │              │   (CSV)      │
    └──────────────┘              └──────────────┘
```

---

## 3. Data Pipeline

### 3.1 Data Loading

**File**: `src/salary_pred.py`, `src/explore_page.py`

```python
# Load dataset
DATA_PATH = 'data/stack-overflow-developer-survey-2025/survey_results_public.csv'
df = pd.read_csv(DATA_PATH, low_memory=False)
```

### 3.2 Data Cleaning

**Steps**:

1. Filter for employed developers
2. Remove rows with missing salary
3. Remove salary outliers ($10k-$500k range)
4. Handle missing values strategically

**Missing Value Strategy**:

- `WorkExp`: Fill with 0 (no professional experience)
- `YearsCode`: Fill with median
- `JobSat`: KNN imputation (k=5)
- `EdLevel`: Fill with mode
- Categorical: 'Missing' indicator

### 3.3 Feature Selection

**22 Core Features**:

```python
selected_features = [
    'MainBranch', 'Age', 'EdLevel', 'Employment', 'WorkExp',
    'LearnCodeChoose', 'LearnCode', 'YearsCode', 'DevType',
    'OrgSize', 'ICorPM', 'RemoteWork', 'Industry', 'Country',
    'JobSat', 'LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith',
    'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith',
    'AIModelsHaveWorkedWith', 'ConvertedCompYearly'
]
```

### 3.4 Data Flow

```
Raw CSV (49,191 rows)
    ↓
Filter Employed (19,500 rows)
    ↓
Remove Missing Salary (17,679 rows)
    ↓
Feature Engineering (267 features)
    ↓
Train-Test Split (80/20)
    ↓
Model Training
```

---

## 4. Model Details

### 4.1 Model Comparison

| Model | Test R² | Test RMSE | Test MAE |
|-------|---------|-----------|----------|
| **XGBoost** | **0.8849** | **$23,120** | **$9,726** |
| Voting Ensemble | 0.7640 | $33,105 | $20,845 |
| Random Forest | 0.6685 | $39,233 | $24,576 |
| KNN | 0.5718 | $44,591 | $24,446 |
| Linear Regression | 0.5209 | $47,163 | $32,127 |
| SVR | -0.0749 | $70,644 | $49,039 |

### 4.2 XGBoost Hyperparameters

**Optimal Configuration** (found via GridSearchCV):

```python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'n_jobs': -1
}
```

**GridSearchCV Settings**:

- Parameter combinations tested: 324
- Cross-validation folds: 3
- Scoring metric: R²
- Total model fits: 972

### 4.3 Model Training Process

```python
# 1. GridSearchCV
xgb_grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

# 2. Fit on training data
xgb_grid.fit(X_train, y_train)

# 3. Extract best model
best_model = xgb_grid.best_estimator_

# 4. Evaluate on test set
predictions = best_model.predict(X_test)
```

### 4.4 Model Persistence

**Saved Artifacts** (`models/saved_model.pkl`):

```python
{
    'model': trained_xgboost_model,
    'model_name': 'XGBoost Regressor',
    'scaler': StandardScaler(),
    'feature_names': list_of_267_features,
    'model_metrics': {
        'R2': 0.8849,
        'RMSE': 23120.26,
        'MAE': 9725.50
    }
}
```

---

## 5. Feature Engineering

### 5.1 Categorical Encoding

**Label Encoding** (ordinal features):

```python
# Age and Education Level
le_age = LabelEncoder()
le_edlevel = LabelEncoder()

dfc['Age'] = le_age.fit_transform(dfc['Age'])
dfc['EdLevel'] = le_edlevel.fit_transform(dfc['EdLevel'])
```

**One-Hot Encoding** (nominal features):

```python
# Categorical columns
categorical_cols = [
    'MainBranch', 'DevType', 'OrgSize', 'ICorPM',
    'RemoteWork', 'Industry', 'Country', 'LearnCodeChoose'
]

dfc = pd.get_dummies(dfc, columns=categorical_cols, drop_first=False)
```

### 5.2 Multi-Select Column Processing

**Technology Columns** (semicolon-separated):

```python
def process_multi_select(df, column_name):
    """
    Convert semicolon-separated values to binary features
    Example: "Python;JavaScript" → Python=1, JavaScript=1
    """
    # Get all unique values
    all_values = df[column_name].dropna().str.split(';')
    unique_values = set([item.strip() for sublist in all_values for item in sublist])
    
    # Create binary columns
    for value in unique_values:
        df[f'{column_name}_{value}'] = df[column_name].str.contains(value, na=False).astype(int)
    
    return df

# Apply to technology columns
tech_columns = [
    'LanguageHaveWorkedWith',
    'DatabaseHaveWorkedWith',
    'PlatformHaveWorkedWith',
    'WebframeHaveWorkedWith',
    'AIModelsHaveWorkedWith',
    'LearnCode'
]
```

**Result**: 178 binary technology features created

### 5.3 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Scale numerical features
cols_to_scale = ['WorkExp', 'YearsCode', 'JobSat']

scaler = StandardScaler()
dfc[cols_to_scale] = scaler.fit_transform(dfc[cols_to_scale])
```

**Note**: CompTotal excluded to prevent data leakage

### 5.4 Final Feature Count

- **Original**: 22 features
- **After encoding**: 267 features
  - Label encoded: 2
  - One-hot encoded: 87
  - Multi-select binary: 178

---

## 6. API Reference

### 6.1 Core Functions

#### `load_data()` - explore_page.py

```python
@st.cache_data
def load_data():
    """
    Load and preprocess the survey data
    
    Returns:
        pd.DataFrame: Preprocessed dataset with outliers removed
    """
```

**Processing Steps**:

1. Load CSV from data folder
2. Rename target column
3. Filter employed developers
4. Handle missing values
5. Remove salary outliers

#### `load_model()` - predict_page.py

```python
def load_model():
    """
    Load the trained model and preprocessing artifacts
    
    Returns:
        dict: Model artifacts including:
            - model: Trained XGBoost model
            - scaler: StandardScaler instance
            - feature_names: List of feature names
            - model_metrics: Performance metrics
    """
```

#### `analyze_multi_select_column()` - explore_page.py

```python
def analyze_multi_select_column(series, top_n=15):
    """
    Analyze semicolon-separated multi-select columns
    
    Args:
        series (pd.Series): Column with semicolon-separated values
        top_n (int): Number of top items to return
    
    Returns:
        pd.DataFrame: Frequency count of items
    """
```

### 6.2 Page Functions

#### `show_predict_page()` - predict_page.py

Main prediction interface with:

- Input form (demographics, experience, tech stack)
- Prediction logic
- Results display

#### `show_explore_page()` - explore_page.py

Data exploration dashboard with:

- 8 major sections
- 40+ interactive visualizations
- Tabbed interfaces for technology analysis

---

## 7. Deployment Guide

### 7.1 Local Deployment

**Prerequisites**:

- Python 3.10+
- pip package manager
- 8GB+ RAM (for model training)

**Steps**:

```bash
# 1. Clone repository
git clone <repository-url>
cd Salary_Predict

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
# Place survey_results_public.csv in data/stack-overflow-developer-survey-2025/

# 5. Train model (optional - model included)
python src/salary_pred.py

# 6. Run application
python run.py
# OR
streamlit run src/app.py
```

### 7.2 Streamlit Cloud Deployment

**Steps**:

1. **Prepare Repository**:

   ```bash
   # Ensure .gitignore includes:
   .venv/
   __pycache__/
   *.pyc
   data/
   .DS_Store
   ```

2. **Create `requirements.txt`**:

   ```
   streamlit==1.52.1
   pandas
   numpy
   scikit-learn==1.7.2
   xgboost
   plotly
   seaborn==0.13.2
   matplotlib
   ```

3. **Deploy**:
   - Push to GitHub
   - Visit [share.streamlit.io](https://share.streamlit.io/)
   - Connect repository
   - Set main file: `src/app.py`
   - Deploy

**Note**: Dataset too large for Streamlit Cloud. Use smaller sample or external storage.

### 7.3 Docker Deployment

**Dockerfile**:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build & Run**:

```bash
docker build -t salary-prediction .
docker run -p 8501:8501 salary-prediction
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### Issue: "Dataset not found"

**Error**:

```
❌ Dataset not found. Please ensure survey_results_public.csv is in the data/stack-overflow-developer-survey-2025 folder.
```

**Solution**:

1. Verify file path: `data/stack-overflow-developer-survey-2025/survey_results_public.csv`
2. Check file permissions
3. Ensure file is not corrupted

#### Issue: "Model file not found"

**Error**:

```
❌ Model file not found. Please run salary_pred.py first to train the model.
```

**Solution**:

```bash
python src/salary_pred.py
```

Wait for training to complete (~40-65 minutes).

#### Issue: Memory Error during training

**Error**:

```
MemoryError: Unable to allocate array
```

**Solution**:

1. Close other applications
2. Reduce GridSearchCV parameter grid
3. Use smaller dataset sample
4. Increase system RAM

#### Issue: Streamlit port already in use

**Error**:

```
OSError: [Errno 98] Address already in use
```

**Solution**:

```bash
# Find process using port 8501
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Linux/Mac

# Kill process
taskkill /PID <PID> /F  # Windows
kill -9 <PID>  # Linux/Mac

# Or use different port
streamlit run src/app.py --server.port 8502
```

### 8.2 Performance Optimization

#### Slow Data Loading

**Solution**:

```python
# Use low_memory=False
df = pd.read_csv(DATA_PATH, low_memory=False)

# Or specify dtypes
dtypes = {
    'WorkExp': 'float32',
    'YearsCode': 'float32',
    # ...
}
df = pd.read_csv(DATA_PATH, dtype=dtypes)
```

#### Slow Predictions

**Solution**:

- Ensure model is cached with `@st.cache_resource`
- Reduce feature count if possible
- Use model.predict() instead of recreating features

### 8.3 Data Quality Issues

#### Missing Values

**Check**:

```python
print(df.isnull().sum())
```

**Handle**:

- Refer to Section 3.2 for missing value strategy
- Adjust imputation methods as needed

#### Outliers

**Detect**:

```python
Q1 = df['AnnualCompUSD'].quantile(0.25)
Q3 = df['AnnualCompUSD'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['AnnualCompUSD'] < Q1 - 1.5*IQR) | (df['AnnualCompUSD'] > Q3 + 1.5*IQR)]
```

**Current Filter**: $10,000 - $500,000

---

## Appendix A: Dependencies

```
streamlit==1.52.1
pandas
numpy
scikit-learn==1.7.2
xgboost
plotly
seaborn==0.13.2
matplotlib
scipy==1.16.3
altair==6.0.0
```

## Appendix B: Dataset Schema

**Source**: Stack Overflow Developer Survey 2025

**Key Columns**:

- `ConvertedCompYearly`: Annual compensation in USD (target)
- `WorkExp`: Years of professional work experience
- `YearsCode`: Years of coding experience
- `Age`: Age range
- `EdLevel`: Education level
- `Country`: Country of residence
- `DevType`: Developer type (multi-select)
- `LanguageHaveWorkedWith`: Programming languages (multi-select)
- And 14 more...

**Total Columns**: 172
**Total Responses**: 49,191

---

**Documentation Version**: 1.0  
**Last Updated**: December 14, 2025  
**Project Version**: 1.0 (GridSearchCV Optimized)
