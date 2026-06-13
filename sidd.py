# ================================
# Predictive Modeling Using ML
# ================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================================
# Load Dataset
# =========================================

# Titanic Dataset from seaborn
df = sns.load_dataset('titanic')

print("First 5 Rows:")
print(df.head())

# =========================================
# Data Preprocessing
# =========================================

# Select important columns
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]

# Remove missing values
df.dropna(inplace=True)

# Convert categorical data to numerical
le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

# Features and Target
X = df.drop('survived', axis=1)
y = df['survived']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================================
# 1. Logistic Regression
# =========================================

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))

# =========================================
# 2. Decision Tree
# =========================================

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))

# =========================================
# 3. Random Forest
# =========================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))

# =========================================
# Best Model Report
# =========================================

print("\n===== Classification Report =====")
print(classification_report(y_test, rf_pred))

# =========================================
# Confusion Matrix
# =========================================

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# ROC Curve
# =========================================

# Probability predictions
rf_probs = rf_model.predict_proba(X_test)[:,1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)

roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(7,6))

plt.plot(fpr, tpr,
         label=f'Random Forest (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.show()

# =========================================
# Feature Importance
# =========================================

importance = rf_model.feature_importances_

feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(feature_df)

# Plot Feature Importance
plt.figure(figsize=(8,5))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df
)

plt.title("Feature Importance")
plt.show()

print("\nProject Completed Successfully!")