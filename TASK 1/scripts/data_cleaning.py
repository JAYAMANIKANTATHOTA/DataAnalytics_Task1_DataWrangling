import pandas as pd
import numpy as np

# Load dataset
file_path = "../data/raw_customer_sales_data.csv"
df = pd.read_csv(file_path)

print("\n========= ORIGINAL DATA =========")
print(df.head())

# ==========================================
# DATA PROFILING
# ==========================================

print("\n========= DATA INFO =========")
print(df.info())

print("\n========= MISSING VALUES =========")
print(df.isnull().sum())

print("\n========= DUPLICATES =========")
print(df.duplicated().sum())

print("\n========= STATISTICS =========")
print(df.describe())

# ==========================================
# DATA CLEANING
# ==========================================

# Remove duplicates

df = df.drop_duplicates()

# Standardize city names

df['city'] = df['city'].str.strip().str.title()

# Standardize dates

df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

# Fill missing revenue with median

revenue_median = df['revenue'].median()
df['revenue'] = df['revenue'].fillna(revenue_median)

# Fill missing profit with median

profit_median = df['profit'].median()
df['profit'] = df['profit'].fillna(profit_median)

# Fill missing DOB with default DOB

default_dob = pd.Timestamp('1998-01-01')
df['dob'] = df['dob'].fillna(default_dob)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

# Customer age

current_year = pd.Timestamp.today().year

df['customer_age'] = current_year - df['dob'].dt.year

# Profit margin percentage

df['profit_margin_percent'] = (
    df['profit'] / df['revenue']
) * 100

# Revenue category

def revenue_category(x):
    if x < 5000:
        return 'Low'
    elif x < 20000:
        return 'Medium'
    else:
        return 'High'


df['revenue_category'] = df['revenue'].apply(revenue_category)

# ==========================================
# OUTLIER IDENTIFICATION
# ==========================================

Q1 = df['revenue'].quantile(0.25)
Q3 = df['revenue'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df['revenue'] < lower_bound) |
    (df['revenue'] > upper_bound)
]

print("\n========= OUTLIERS =========")
print(outliers)

# ==========================================
# EXPORT CLEAN DATASET
# ==========================================

output_path = "../data/cleaned_customer_sales_data.csv"

df.to_csv(output_path, index=False)

print("\n========= CLEANED DATA =========")
print(df.head())

print("\nCleaned dataset exported successfully.")