import pandas as pd

# Read the data
df = pd.read_csv('data/data.csv')

# Print basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Print statistics for numerical columns
print("\nNumerical Columns Statistics:")
print(df[['floor_area_sqm', 'resale_price']].describe())

# Print value counts for categorical columns
print("\nCategorical Columns Value Counts:")
categorical_cols = ['flat_type', 'storey_range', 'town_name']
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts().head())

# Print missing values
print("\nMissing Values:")
print(df.isna().sum()) 