# Machine Learning & Data Science Workspace 🚀

Welcome to my Machine Learning and Data Science repository! This repository serves as a structured, hands-on portfolio showcasing Python programming fundamentals, core data science libraries, classic machine learning algorithms (regression, classification, ensemble methods, unsupervised learning, anomaly detection), and end-to-end predictive modeling projects.

---

## 📂 Repository Structure

The workspace is organized into logical directories reflecting a progressive learning path and project-based applications:

| Directory / File | Description | Key Topics / Notebooks |
| :--- | :--- | :--- |
| 📁 **`Python_Basics/`** & **`PythonLib/`** | Foundations of Python & Data Science Stack | Basic Syntax, Data Structures, Control Flow, Functions, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn** |
| 📁 **`Regression/`** | Continuous value prediction models | Linear Regression, Lasso & Ridge Regression, LassoCV, Decision Tree Regressor, Insurance Charges Analysis |
| 📁 **`Classification/`** | Categorical value prediction models | Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree Classifier, Support Vector Machines (SVM) |
| 📁 **`Unsupervised/`** | Clustering and dimensionality reduction | K-Means, DBSCAN, Hierarchical Clustering, PCA |
| 📁 **`Ensemble/`** | Ensemble machine learning methods | Random Forest |
| 📁 **`Anomaly_Detection/`** | Identifying rare items, events or observations | Isolation Forest |
| 📁 **`evaluation_matrix/`** | Model evaluation metrics | Accuracy, Precision |
---

## 🛠️ Detailed Component Breakdown

### 1. Python Foundations & Core Libraries (`Python_Basics/` & `PythonLib/`)
* **Python Basics**: Four structured notebooks covering basic syntax, data structures (lists, dicts, tuples, sets), control flow statements, and function design.
* **NumPy (`numpy.ipynb`)**: Multi-dimensional arrays, vectorization, indexing, slicing, and mathematical operations.
* **Pandas (`pandas.ipynb`)**: Series, DataFrames, data cleaning, filtering, sorting, group-by operations, handling missing data, and merging datasets.
* **Matplotlib & Seaborn (`matplotlib.ipynb` & `seaborn.ipynb`)**: Custom visualizations, line plots, scatter plots, bar charts, histograms, heatmaps, box plots, and pair plots.

### 2. Regression Analysis (`Regression/`)
* **Linear Regression**: Implementation of simple and multiple linear regression (e.g., predicting insurance charges using `15_insurance.csv`).
* **Regularization**: Ridge and Lasso regression implementations including `LassoCV` for cross-validated feature selection.
* **Non-Linear Regression**: Non-linear mapping using Decision Tree Regressors.

### 3. Classification Algorithms (`Classification/`)
* Hands-on implementation of the four foundational classification algorithms:
  * **Logistic Regression**: Linear boundary classifier.
  * **K-Nearest Neighbors (KNN)**: Distance-based classification.
  * **Decision Trees**: Entropy/Gini-based splits.
  * **Support Vector Machines (SVM)**: Margins and kernel tricks.

### 4. Unsupervised Learning (`Unsupervised/`)
* **K-Means**: Centroid-based clustering.
* **DBSCAN**: Density-based spatial clustering.
* **Hierarchical Clustering**: Agglomerative tree-based clustering.
* **PCA (Principal Component Analysis)**: Dimensionality reduction.

### 5. Ensemble Methods (`Ensemble/`)
* **Random Forest**: Ensemble of decision trees for robust predictions.

### 6. Anomaly Detection (`Anomaly_Detection/`)
* **Isolation Forest**: Unsupervised approach for identifying outliers and anomalies.

### 7. Evaluation Metrics (`evaluation_matrix/`)
* **Classification Metrics**: Practical implementations of accuracy and precision.

---

## 🏆 Key Projects Highlight

### 🛳️ Titanic Survival Prediction
* **Scripts**: [titanic_project.py](file:///c:/Users/Siddharth/JupyterNoteBook/titanic_project.py) (Data cleaning & feature engineering) & [sidd.py](file:///c:/Users/Siddharth/JupyterNoteBook/sidd.py) (Machine learning model evaluation).
* **Workflow**:
  * Missing values handled (Age filled with median, Deck dropped, Embarked filled with mode).
  * Feature engineering: Created `family_size` and `age_group` variables.
  * Comparative analysis: Trained **Logistic Regression**, **Decision Tree**, and **Random Forest**.
  * Evaluation: Evaluated via Accuracy score, Classification Reports, Confusion Matrix heatmaps, **ROC-AUC Curves**, and **Feature Importance** bar plots.

### 📈 Sales Prediction Model
* **Script**: [Real-world Data Project.py](file:///c:/Users/Siddharth/JupyterNoteBook/Real-world%20Data%20Project.py).
* **Details**: Builds a multiple linear regression model using features `Advertising` and `Store_Size` from `sales.csv` to predict sales performance, evaluating predictions with Mean Absolute Error (MAE) and R² Score, and plotting Actual vs. Predicted values.

### 📊 Reusable EDA Script
* **Script**: [EDA_PROJECT.py](file:///c:/Users/Siddharth/JupyterNoteBook/EDA_PROJECT.py).
* **Details**: A template pipeline script that automatically analyzes any `data.csv`:
  * Fills missing numeric features with the median and categorical features with the mode.
  * Generates histograms, boxplots, pairplots, and correlation heatmaps.
  * Identifies outliers in numeric columns using the Interquartile Range (IQR) method.

---

## 🚀 How to Set Up and Run Locally

Follow these steps to run the scripts and notebooks on your local machine:

### 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd JupyterNoteBook
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
# Create environment
python -m venv myenv

# Activate environment
# On Windows (Command Prompt)
myenv\Scripts\activate
# On Windows (PowerShell)
.\myenv\Scripts\Activate.ps1
# On macOS/Linux
source myenv/bin/activate
```

### 3. Install Dependencies
Create a `requirements.txt` containing the necessary packages or run:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

### 4. Run Jupyter Notebooks
```bash
jupyter notebook
```

---

## 💻 Technologies & Tools Used
* **Languages**: Python 3.x
* **Core Libraries**: NumPy, Pandas, Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn, XGBoost
* **Environment**: Jupyter Notebook / Lab, VS Code
