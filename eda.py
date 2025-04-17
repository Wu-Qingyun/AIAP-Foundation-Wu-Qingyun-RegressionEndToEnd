#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis for HDB Flat Resale Price Prediction
# 
# This notebook contains comprehensive exploratory data analysis of the HDB flat resale price dataset.

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import yaml
from scipy import stats
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'eda_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)

# ## 1. Load and Examine Data

def load_data():
    """Load data from CSV file using configuration."""
    try:
        with open('src/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        df = pd.read_csv(config['data']['train_path'])
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def check_and_convert_data_types(df):
    """Check data types and convert non-numeric columns to appropriate types."""
    logging.info("Checking and converting data types...")
    
    # Display initial data types
    logging.info("\nInitial data types:")
    logging.info(df.dtypes)
    
    # Convert remaining_lease to numeric (extract numbers from string)
    df['remaining_lease'] = pd.to_numeric(df['remaining_lease'].str.extract(r'(\d+)')[0])
    
    # Convert block to numeric (extract numbers)
    df['block_number'] = pd.to_numeric(df['block'].str.extract(r'(\d+)')[0], errors='coerce')
    
    # Convert month to datetime
    df['month'] = pd.to_datetime(df['month'])
    
    logging.info("\nData types after conversion:")
    logging.info(df.dtypes)
    
    return df

def check_duplicates(df):
    """Check for and handle duplicate rows."""
    logging.info("Checking for duplicate rows...")
    
    duplicates = df.duplicated().sum()
    logging.info(f"Number of duplicate rows: {duplicates}")
    
    if duplicates > 0:
        df = df.drop_duplicates()
        logging.info(f"Dropped {duplicates} duplicate rows. New shape: {df.shape}")
    
    return df

def analyze_missing_values(df):
    """Analyze and handle missing values."""
    logging.info("Analyzing missing values...")
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    
    logging.info("\nMissing values analysis:")
    logging.info(missing_info)
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    logging.info("Missing values handled using mode for categorical and median for numerical columns")
    return df

def compute_detailed_statistics(df):
    """Compute detailed statistics for numerical columns."""
    logging.info("Computing detailed statistics...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    stats_dict = {}
    for col in numerical_cols:
        stats_dict[col] = {
            'count': df[col].count(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'mode': df[col].mode()[0],
            'std': df[col].std(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    stats_df = pd.DataFrame(stats_dict).T
    logging.info("\nDetailed statistics:")
    logging.info(stats_df)
    
    return stats_df

def identify_outliers(df):
    """Identify outliers using IQR and Z-score methods."""
    logging.info("Identifying outliers...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    
    for col in numerical_cols:
        # IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(df[col]))
        zscore_outliers = df[z_scores > 3][col]
        
        outlier_info[col] = {
            'IQR_outliers_count': len(iqr_outliers),
            'Zscore_outliers_count': len(zscore_outliers),
            'IQR_outliers_percentage': (len(iqr_outliers) / len(df)) * 100,
            'Zscore_outliers_percentage': (len(zscore_outliers) / len(df)) * 100
        }
    
    outlier_df = pd.DataFrame(outlier_info).T
    logging.info("\nOutlier analysis:")
    logging.info(outlier_df)
    
    return outlier_df

def analyze_distributions(df):
    """Analyze distributions and identify skewed features."""
    logging.info("Analyzing distributions...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    skewed_features = {}
    
    for col in numerical_cols:
        skewness = df[col].skew()
        if abs(skewness) > 1:  # Consider features with |skewness| > 1 as skewed
            skewed_features[col] = skewness
    
    logging.info("\nSkewed features (|skewness| > 1):")
    logging.info(skewed_features)
    
    return skewed_features

def apply_log_transformation(df, skewed_features):
    """Apply log transformation to skewed features."""
    logging.info("Applying log transformation to skewed features...")
    
    df_transformed = df.copy()
    for col in skewed_features.keys():
        if df[col].min() > 0:  # Only apply log transformation to positive values
            df_transformed[f'{col}_log'] = np.log1p(df[col])
    
    return df_transformed

def analyze_feature_relationships(df):
    """Analyze relationships between features."""
    logging.info("Analyzing feature relationships...")
    
    # Correlation analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    logging.info("Correlation heatmap saved as 'correlation_heatmap.png'")
    
    return correlation_matrix

def main():
    """Main function to run the EDA pipeline."""
    try:
        # Load data
        df = load_data()
        
        # Check and convert data types
        df = check_and_convert_data_types(df)
        
        # Check for duplicates
        df = check_duplicates(df)
        
        # Analyze and handle missing values
        df = analyze_missing_values(df)
        
        # Compute detailed statistics
        stats_df = compute_detailed_statistics(df)
        
        # Identify outliers
        outlier_df = identify_outliers(df)
        
        # Analyze distributions
        skewed_features = analyze_distributions(df)
        
        # Apply log transformation to skewed features
        df_transformed = apply_log_transformation(df, skewed_features)
        
        # Analyze feature relationships
        correlation_matrix = analyze_feature_relationships(df_transformed)
        
        logging.info("EDA completed successfully!")
        
        return df_transformed, stats_df, outlier_df, correlation_matrix
        
    except Exception as e:
        logging.error(f"Error in EDA pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# ## 2. Data Quality Assessment

# Check for missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

print("Missing values per column:")
for col, count, percentage in zip(df.columns, missing_values, missing_percentage):
    print(f"{col}: {count} ({percentage:.2f}%)")

# Visualize missing values
plt.figure(figsize=(12, 6))
sns.barplot(x=df.columns, y=missing_percentage)
plt.title('Missing Values Percentage by Column')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Percentage')
plt.tight_layout()
plt.show()

# ## 3. Target Variable Analysis

target_col = config['features']['target_column']

# Distribution of target variable
plt.figure(figsize=(12, 6))
sns.histplot(df[target_col], kde=True)
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Count')
plt.show()

# Box plot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(y=df[target_col])
plt.title(f'Box Plot of {target_col}')
plt.show()

# Summary statistics of target variable
print(f"Summary statistics of {target_col}:")
print(df[target_col].describe())

# ## 4. Numerical Features Analysis

numerical_cols = config['features']['numerical_columns']

# Correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_cols + [target_col]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Pair plot for numerical features
sns.pairplot(df[numerical_cols + [target_col]], diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features', y=1.02)
plt.show()

# ## 5. Categorical Features Analysis

categorical_cols = config['features']['categorical_columns']

# Count plots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    value_counts = df[col].value_counts()
    
    # If too many categories, show top 10
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
    
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Box plots for categorical features vs target
for col in categorical_cols:
    plt.figure(figsize=(12, 6))
    
    # If too many categories, show top 10
    if df[col].nunique() > 10:
        top_categories = df[col].value_counts().head(10).index
        df_filtered = df[df[col].isin(top_categories)]
    else:
        df_filtered = df
    
    sns.boxplot(x=col, y=target_col, data=df_filtered)
    plt.title(f'{target_col} by {col}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# ## 6. Feature Engineering Insights

# Convert lease_commence_date to datetime
df['lease_commence_date'] = pd.to_datetime(df['lease_commence_date'])

# Calculate age of the flat
current_year = pd.Timestamp.now().year
df['flat_age'] = current_year - df['lease_commence_date'].dt.year

# Calculate price per square meter
df['price_per_sqm'] = df[target_col] / df['floor_area_sqm']

# Plot flat age vs resale price
plt.figure(figsize=(12, 6))
sns.scatterplot(x='flat_age', y=target_col, data=df, alpha=0.5)
plt.title('Resale Price vs Flat Age')
plt.xlabel('Flat Age (years)')
plt.ylabel(target_col)
plt.show()

# Plot price per square meter distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['price_per_sqm'], kde=True)
plt.title('Distribution of Price per Square Meter')
plt.xlabel('Price per Square Meter')
plt.ylabel('Count')
plt.show()

# ## 7. Summary of Findings

# ### Key Insights:
# 
# 1. **Data Quality**:
#    - [To be filled after running the notebook]
# 
# 2. **Target Variable (Resale Price)**:
#    - [To be filled after running the notebook]
# 
# 3. **Feature Relationships**:
#    - [To be filled after running the notebook]
# 
# 4. **Feature Engineering Opportunities**:
#    - [To be filled after running the notebook]
# 
# 5. **Model Selection Considerations**:
#    - [To be filled after running the notebook] 