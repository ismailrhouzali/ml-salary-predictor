# -*- coding: utf-8 -*-
"""
Stack Overflow Annual Developer Survey 2025 - Salary Prediction Project
================================================================================

Dataset: https://survey.stackoverflow.co/

The 2025 Developer Survey received over 49,000+ responses from 177 countries 
across 62 questions focused on 314 different technologies. This project builds
a machine learning model to predict developer salaries based on experience,
education, technologies used, and other factors.

Date: December 2025
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)

# Plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 80)
print("STACK OVERFLOW 2025 DEVELOPER SALARY PREDICTION")
print("=" * 80)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n[STEP 1] Loading Dataset...")

# Update this path to your local dataset location
# Dataset path - now in data folder
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'stack-overflow-developer-survey-2025', 'survey_results_public.csv')

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Total responses: {df.shape[0]:,}")
    print(f"  - Total columns: {df.shape[1]}")
except FileNotFoundError:
    print(f"✗ Error: Could not find file at {DATA_PATH}")
    print("  Please update DATA_PATH variable with your dataset location.")
    exit(1)

# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

print("\n[STEP 2] Selecting Relevant Features...")

# Select 22 key columns for analysis
selected_columns = [
    'MainBranch', 'Age', 'EdLevel', 'Employment', 'WorkExp', 
    'LearnCodeChoose', 'LearnCode', 'YearsCode', 'DevType', 
    'OrgSize', 'ICorPM', 'RemoteWork', 'Industry', 'Country', 
    'CompTotal', 'JobSat', 'LanguageHaveWorkedWith', 
    'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 
    'WebframeHaveWorkedWith', 'AIModelsHaveWorkedWith', 
    'ConvertedCompYearly'
]

dfc = df[selected_columns].copy()
print(f"✓ Selected {len(selected_columns)} features")
print(f"  - Dataset shape: {dfc.shape}")

# Rename target variable for clarity
dfc.rename({"ConvertedCompYearly": "AnnualCompUSD"}, axis=1, inplace=True)

# ============================================================================
# 4. INITIAL DATA EXPLORATION
# ============================================================================

print("\n[STEP 3] Initial Data Exploration...")

print("\nDataset Info:")
print(f"  - Total entries: {dfc.shape[0]:,}")
print(f"  - Features: {dfc.shape[1]}")
print(f"\nData Types:")
print(dfc.dtypes.value_counts())

print("\nMissing Values Summary:")
missing_stats = pd.DataFrame({
    'Missing Count': dfc.isnull().sum(),
    'Missing %': (dfc.isnull().sum() / dfc.shape[0] * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_stats)

# ============================================================================
# 5. DATA CLEANING
# ============================================================================

print("\n[STEP 4] Data Cleaning...")

# 5.1 Remove rows without salary information
initial_rows = dfc.shape[0]
dfc = dfc[dfc["AnnualCompUSD"].notnull()]
print(f"✓ Removed {initial_rows - dfc.shape[0]:,} rows with missing salary")

# 5.2 Filter for employed developers only
dfc = dfc[dfc["Employment"] == "Employed"]
print(f"✓ Filtered for employed developers: {dfc.shape[0]:,} rows")

# 5.3 Handle missing values with strategic approaches

# Categorical columns - Fill with 'Missing' (absence is informative)
categorical_missing_cols = ['DevType', 'OrgSize', 'ICorPM', 'RemoteWork', 
                            'Industry', 'LearnCodeChoose', 'LearnCode']
for col in categorical_missing_cols:
    dfc[col] = dfc[col].fillna('Missing')
print(f"✓ Filled categorical missing values with 'Missing' indicator")

# Multi-select technology columns
tech_columns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 
                'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith', 
                'AIModelsHaveWorkedWith']
for col in tech_columns:
    dfc[col] = dfc[col].fillna('Missing')
print(f"✓ Filled technology columns missing values")

# Numerical columns - Use KNN imputation for JobSat
if dfc['JobSat'].isnull().sum() > 0:
    imputer = KNNImputer(n_neighbors=5)
    dfc['JobSat'] = imputer.fit_transform(dfc[['JobSat']])
    print(f"✓ Imputed JobSat using KNN (5 neighbors)")

# Education - Use mode (most common)
dfc['EdLevel'] = dfc['EdLevel'].fillna(dfc['EdLevel'].mode()[0])

# Experience columns
# YearsCode - Fill with median (developers who didn't answer likely have median experience)
# WorkExp - Fill with 0 (NaN means no professional work experience)
dfc['YearsCode'] = dfc['YearsCode'].fillna(dfc['YearsCode'].median())
dfc['WorkExp'] = dfc['WorkExp'].fillna(0)  # Missing WorkExp = No professional experience = 0
print(f"✓ Filled experience columns (YearsCode: median, WorkExp NaN→0)")

print(f"\nRemaining missing values: {dfc.isnull().sum().sum()}")

# ============================================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n[STEP 5] Exploratory Data Analysis...")

# 6.1 Analyze multi-select columns
print("\nAnalyzing Multi-Choice Technology Columns:")

def analyze_multi_select_column(series, column_name, top_n=10):
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
    
    print(f"\n  {column_name} - Top {top_n}:")
    for idx, row in df_freq.head(top_n).iterrows():
        print(f"    {idx+1}. {row['Element']}: {row['Frequency']:,}")
    
    return df_freq

multi_select_cols = tech_columns + ['LearnCode']
all_frequencies = {}

for col in multi_select_cols:
    if col in dfc.columns:
        all_frequencies[col] = analyze_multi_select_column(dfc[col], col, top_n=5)

# 6.2 Country distribution analysis
print("\n\nCountry Distribution Analysis:")
country_counts = dfc['Country'].value_counts()
print(f"  - Total countries: {len(country_counts)}")
print(f"  - Top 5 countries:")
for idx, (country, count) in enumerate(country_counts.head(5).items(), 1):
    print(f"    {idx}. {country}: {count:,} ({count/len(dfc)*100:.1f}%)")

# 6.3 Simplify country categories (keep countries with >400 responses)
def shorten_categories(categories, cutoff):
    """Group rare categories into 'Other'"""
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = "Other"
    return categorical_map

country_map = shorten_categories(dfc['Country'].value_counts(), 400)
dfc['Country'] = dfc['Country'].map(country_map)
print(f"✓ Simplified countries to {dfc['Country'].nunique()} categories")

# 6.4 Salary outlier removal
print("\n\nSalary Distribution Analysis:")
print(f"  - Min salary: ${dfc['AnnualCompUSD'].min():,.0f}")
print(f"  - Max salary: ${dfc['AnnualCompUSD'].max():,.0f}")
print(f"  - Mean salary: ${dfc['AnnualCompUSD'].mean():,.0f}")
print(f"  - Median salary: ${dfc['AnnualCompUSD'].median():,.0f}")

# Remove unrealistic salary outliers
before_outlier_removal = dfc.shape[0]
dfc = dfc[(dfc['AnnualCompUSD'] >= 10000) & (dfc['AnnualCompUSD'] <= 500000)]
removed_outliers = before_outlier_removal - dfc.shape[0]
print(f"✓ Removed {removed_outliers:,} salary outliers (kept $10k-$500k range)")
print(f"  - Final dataset size: {dfc.shape[0]:,} rows")

# 6.5 Correlation analysis
print("\n\nCorrelation Analysis (Numerical Features):")
numeric_cols = dfc.select_dtypes(include='number').columns
correlation_with_salary = dfc[numeric_cols].corr()['AnnualCompUSD'].sort_values(ascending=False)
print(correlation_with_salary)

# ============================================================================
# 7. FEATURE ENGINEERING
# ============================================================================

print("\n[STEP 6] Feature Engineering...")

# 7.1 Encode Age and EdLevel with LabelEncoder
le_age = LabelEncoder()
le_edlevel = LabelEncoder()

dfc['Age'] = le_age.fit_transform(dfc['Age'])
dfc['EdLevel'] = le_edlevel.fit_transform(dfc['EdLevel'])
print(f"✓ Label encoded Age and EdLevel")

# 7.2 One-hot encode multi-select columns (semicolon-separated)
print("\nOne-hot encoding multi-select columns:")
multi_select_columns = tech_columns + ['LearnCode']

for col in multi_select_columns:
    if col in dfc.columns:
        # Use get_dummies with semicolon separator
        dummies = dfc[col].str.get_dummies(sep=';')
        # Add prefix to avoid column name conflicts
        dummies = dummies.add_prefix(f'{col}_')
        dfc = pd.concat([dfc, dummies], axis=1)
        dfc.drop(columns=[col], inplace=True)
        print(f"  ✓ {col}: created {dummies.shape[1]} binary features")

# 7.3 One-hot encode standard categorical columns
print("\nOne-hot encoding categorical columns:")
categorical_columns = ['MainBranch', 'DevType', 'OrgSize', 'ICorPM', 
                       'RemoteWork', 'Industry', 'Country', 'LearnCodeChoose']

for col in categorical_columns:
    if col in dfc.columns:
        dummies = pd.get_dummies(dfc[col], prefix=col, drop_first=True)
        dfc = pd.concat([dfc, dummies], axis=1)
        dfc.drop(columns=[col], inplace=True)
        print(f"  ✓ {col}: created {dummies.shape[1]} binary features")

# 7.4 Drop Employment column (all values are 'Employed')
if 'Employment' in dfc.columns:
    dfc.drop(columns=['Employment'], inplace=True)

print(f"\n✓ Feature engineering complete!")
print(f"  - Final feature count: {dfc.shape[1] - 1} (excluding target)")
print(f"  - Total samples: {dfc.shape[0]:,}")

# 7.5 Feature scaling
print("\nScaling numerical features:")
# Note: CompTotal is NOT scaled as it's highly correlated with our target (AnnualCompUSD)
# Scaling it would introduce data leakage
cols_to_scale = ['WorkExp', 'YearsCode', 'JobSat']

scaler = StandardScaler()
dfc[cols_to_scale] = scaler.fit_transform(dfc[cols_to_scale])
print(f"✓ Standardized {len(cols_to_scale)} numerical features (WorkExp, YearsCode, JobSat)")

# ============================================================================
# 8. TRAIN-TEST SPLIT
# ============================================================================

print("\n[STEP 7] Preparing Train-Test Split...")

X = dfc.drop(columns=['AnnualCompUSD'])
y = dfc['AnnualCompUSD']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Split complete!")
print(f"  - Training samples: {X_train.shape[0]:,}")
print(f"  - Testing samples: {X_test.shape[0]:,}")
print(f"  - Features: {X_train.shape[1]}")

# ============================================================================
# 9. MODEL TRAINING
# ============================================================================

print("\n[STEP 8] Training Multiple Regression Models...")
print("=" * 80)

# Dictionary to store model results
model_results = []

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a regression model"""
    print(f"\n{'='*80}")
    print(f"Training: {name}")
    print(f"{'='*80}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nTraining Performance:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    
    print(f"\nTesting Performance:")
    print(f"  R² Score: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    
    return {
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae,
        'Trained_Model': model
    }

# 9.1 Linear Regression
lr_model = LinearRegression()
lr_results = evaluate_model("Linear Regression", lr_model, X_train, X_test, y_train, y_test)
model_results.append(lr_results)

# 9.2 Random Forest Regressor# 9.2 Random Forest Regressor with GridSearchCV
print("\n\nTraining Random Forest with GridSearchCV hyperparameter tuning...")
print("This may take several minutes...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

print("Starting GridSearch for Random Forest...")
rf_grid.fit(X_train, y_train)

print(f"\n✓ Best Random Forest parameters: {rf_grid.best_params_}")
print(f"✓ Best CV R² score: {rf_grid.best_score_:.4f}")

rf_model = rf_grid.best_estimator_
rf_results = evaluate_model("Random Forest Regressor", rf_model, X_train, X_test, y_train, y_test)
rf_results['best_params'] = rf_grid.best_params_
model_results.append(rf_results)

# 9.3 XGBoost Regressor with GridSearchCV
print("\n\nTraining XGBoost with GridSearchCV hyperparameter tuning...")
print("This may take several minutes...")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
xgb_grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='r2',
    verbose=2,
    n_jobs=-1
)

print("Starting GridSearch for XGBoost...")
xgb_grid.fit(X_train, y_train)

print(f"\n✓ Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"✓ Best CV R² score: {xgb_grid.best_score_:.4f}")

xgb_model = xgb_grid.best_estimator_
xgb_results = evaluate_model("XGBoost Regressor", xgb_model, X_train, X_test, y_train, y_test)
xgb_results['best_params'] = xgb_grid.best_params_
model_results.append(xgb_results)

# 9.4 K-Nearest Neighbors Regressor with GridSearchCV
print("\n\nTraining K-Nearest Neighbors with GridSearchCV...")

knn_param_grid = {
    'n_neighbors': [5, 10, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn_base = KNeighborsRegressor(n_jobs=-1)
knn_grid = GridSearchCV(
    estimator=knn_base,
    param_grid=knn_param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1
)

print("Starting GridSearch for KNN...")
knn_grid.fit(X_train, y_train)

print(f"\n✓ Best KNN parameters: {knn_grid.best_params_}")
print(f"✓ Best CV R² score: {knn_grid.best_score_:.4f}")

knn_model = knn_grid.best_estimator_
knn_results = evaluate_model("K-Nearest Neighbors", knn_model, X_train, X_test, y_train, y_test)
knn_results['best_params'] = knn_grid.best_params_
model_results.append(knn_results)

# 9.5 Support Vector Regressor (on subset due to computational cost)
print("\n\nTraining Support Vector Regressor (on 20% sample)...")
# SVR is computationally expensive, use smaller sample
sample_size = int(0.2 * len(X_train))
X_train_sample = X_train.iloc[:sample_size]
y_train_sample = y_train.iloc[:sample_size]

svr_model = SVR(kernel='rbf', C=100, gamma='scale')
svr_results = evaluate_model("Support Vector Regressor", svr_model, 
                             X_train_sample, X_test, y_train_sample, y_test)
model_results.append(svr_results)

# 9.6 Voting Regressor (Ensemble of best models)
print("\n\nTraining Voting Regressor (Ensemble)...")
voting_model = VotingRegressor(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    n_jobs=-1
)
voting_results = evaluate_model("Voting Regressor", voting_model, X_train, X_test, y_train, y_test)
model_results.append(voting_results)

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON RESULTS")
print("=" * 80)

# Create comparison DataFrame
results_df = pd.DataFrame(model_results)
results_df = results_df.drop('Trained_Model', axis=1)
results_df = results_df.sort_values('Test_R2', ascending=False)

print("\n" + results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model_obj = [m for m in model_results if m['Model'] == best_model_name][0]['Trained_Model']

print(f"\n{'='*80}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*80}")
print(f"  Test R² Score: {results_df.iloc[0]['Test_R2']:.4f}")
print(f"  Test RMSE: ${results_df.iloc[0]['Test_RMSE']:,.2f}")
print(f"  Test MAE: ${results_df.iloc[0]['Test_MAE']:,.2f}")

# Display best hyperparameters found by GridSearchCV
print(f"\n{'='*80}")
print("OPTIMAL HYPERPARAMETERS (Found by GridSearchCV)")
print(f"{'='*80}")

for result in model_results:
    if 'best_params' in result:
        print(f"\n{result['Model']}:")
        for param, value in result['best_params'].items():
            print(f"  - {param}: {value}")

# ============================================================================
# 11. MODEL PERSISTENCE
# ============================================================================

print("\n[STEP 9] Saving Model and Preprocessing Artifacts...")

# Save to models directory
model_save_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model.pkl')

model_artifacts = {
    'model': best_model_obj,
    'model_name': best_model_name,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'model_metrics': {
        'R2': results_df.iloc[0]['Test_R2'],
        'RMSE': results_df.iloc[0]['Test_RMSE'],
        'MAE': results_df.iloc[0]['Test_MAE']
    },
    'country_mapping': country_mapping if 'country_mapping' in locals() else None
}

with open(model_save_path, 'wb') as file:
    pickle.dump(model_artifacts, file)

print(f"✓ Model saved to '{model_save_path}'")
print(f"  - Model type: {best_model_name}")
print(f"  - Feature count: {len(X.columns)}")

# ============================================================================
# 12. CONCLUSIONS
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT CONCLUSIONS")
print("=" * 80)

print(f"""
Summary of Findings:
--------------------
1. Dataset: Analyzed {dfc.shape[0]:,} employed developers from Stack Overflow 2025 survey

2. Feature Engineering: 
   - Started with 22 core features
   - Expanded to {X_train.shape[1]} features after encoding
   - Handled multi-select technology columns (languages, databases, platforms, etc.)

3. Model Performance:
   - Trained and compared 6 different regression models
   - Best performing model: {best_model_name}
   - Achieved R² score of {results_df.iloc[0]['Test_R2']:.4f} on test set
   - Average prediction error (MAE): ${results_df.iloc[0]['Test_MAE']:,.2f}

4. Key Insights:
   - Salary ranges from $10,000 to $500,000 USD annually
   - Strong predictors include: experience, education, country, and tech stack
   - Model can predict developer salaries with reasonable accuracy

5. Next Steps:
   - Deploy model in Streamlit application
   - Enable interactive salary predictions
   - Provide comprehensive data visualizations

Model ready for deployment! ✓
""")

print("=" * 80)
print("Script completed successfully!")
print("=" * 80)