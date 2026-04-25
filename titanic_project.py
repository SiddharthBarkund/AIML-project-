import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# load dataset
df = sns.load_dataset("titanic")

# fill missing values
df["age"].fillna(df["age"].median(), inplace=True)
df.drop("deck", axis=1, inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# remove duplicates
df.drop_duplicates(inplace=True)

# feature engineering
df["family_size"] = df["sibsp"] + df["parch"] + 1

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["Child", "Teen", "Adult", "MidAge", "Senior"],
)

# visualization
plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
sns.countplot(x="survived", data=df)

plt.subplot(2, 3, 2)
sns.countplot(x="sex", hue="survived", data=df)

plt.subplot(2, 3, 3)
sns.countplot(x="pclass", hue="survived", data=df)

plt.subplot(2, 3, 4)
sns.histplot(df["fare"], bins=30, kde=True)

plt.subplot(2, 3, 5)
sns.histplot(df["age"], bins=30, kde=True)

plt.subplot(2, 3, 6)
sns.heatmap(df.corr(numeric_only=True), annot=True)

plt.tight_layout()
plt.show()

# save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)
