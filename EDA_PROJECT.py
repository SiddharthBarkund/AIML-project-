# Exploratory Data Analysis (EDA) Project - Full Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# Replace 'data.csv' with your dataset file
df = pd.read_csv("data.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Statistical Summary
# -----------------------------
print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# Handling Missing Values
# -----------------------------
# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Correlation Heatmap
# -----------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Histograms
# -----------------------------
df.hist(figsize=(12, 10))
plt.suptitle("Histograms")
plt.show()

# -----------------------------
# Boxplots
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# -----------------------------
# Pairplot
# -----------------------------
sns.pairplot(df[numeric_cols])
plt.show()

# -----------------------------
# Countplots for Categorical Columns
# -----------------------------
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[col])
    plt.title(f"Countplot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# -----------------------------
# Outlier Detection
# -----------------------------
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nOutliers in {col}: {len(outliers)}")

# -----------------------------
# Unique Values
# -----------------------------
for col in df.columns:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())

# -----------------------------
# Final Insights
# -----------------------------
print("\nEDA Completed Successfully!")
print("Insights:")
print("- Checked missing values")
print("- Generated statistical summaries")
print("- Found correlations")
print("- Visualized distributions")
print("- Detected outliers")
print("- Analyzed categorical features")
