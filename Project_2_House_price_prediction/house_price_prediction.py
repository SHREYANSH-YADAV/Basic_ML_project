# # House Price Prediction System
# AIML Summer Internship 2026 — IIHMF, MNNIT Allahabad
# 
# Capstone Notebook: full ML lifecycle — problem understanding, data preprocessing, EDA, feature engineering, model building, evaluation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
import joblib, json, warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# ## Phase 2: Data Collection
# Dataset loaded from the `Dataset/` folder.

df = pd.read_csv("house_price_dataset.csv")
print("Shape:", df.shape)
df.head()

df.info()

df.describe(include='all').T

# ## Phase 3: Data Preprocessing
# Handle missing values, remove duplicates, treat outliers, encode categorical features.

print("Missing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

dup_count = df.duplicated(subset=[c for c in df.columns if c != 'HouseID']).sum()
print("\nDuplicate rows:", dup_count)
df = df.drop_duplicates(subset=[c for c in df.columns if c != 'HouseID']).reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

numeric_cols = ['Area_SqFt', 'Bedrooms', 'Bathrooms', 'Age_Years', 'Amenities_Count', 'Distance_From_City_Center_KM', 'Parking_Spaces']
categorical_cols = ['Location_Tier', 'Furnishing_Status']

# Impute numeric with median, categorical with mode
for c in numeric_cols:
    df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

print("Missing values after imputation:", df[numeric_cols + categorical_cols].isnull().sum().sum())

# Outlier treatment via IQR capping on numeric columns
for c in numeric_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < low) | (df[c] > high)).sum()
    df[c] = df[c].clip(low, high)
    print(f"{c}: {n_out} outliers capped")

# ## Phase 4: Exploratory Data Analysis (EDA)
# Univariate, bivariate, and correlation analysis with visualizations.

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), numeric_cols[:6]):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['Price_Lakhs'], kde=True, color="darkorange")
plt.title("Target Distribution: Price_Lakhs")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4))
sns.boxplot(data=df[numeric_cols[:4]], ax=axes[0])
axes[0].set_title("Boxplots (outlier check, first 4 numeric cols)")
axes[0].tick_params(axis='x', rotation=30)
if len(categorical_cols) > 0:
    sns.countplot(x=df[categorical_cols[0]], ax=axes[1], palette="pastel")
    axes[1].set_title(f"Category counts: {categorical_cols[0]}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,7))
corr = df[numeric_cols + (["Price_Lakhs"] if 'regression'=='regression' else [])].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(df[numeric_cols[0]], df['Price_Lakhs'] if 'regression'=='regression' else df[numeric_cols[1]], alpha=0.4, color="teal")
plt.xlabel(numeric_cols[0])
plt.ylabel('Price_Lakhs' if 'regression'=='regression' else numeric_cols[1])
plt.title(f"Scatter: {numeric_cols[0]} vs " + ('Price_Lakhs' if 'regression'=='regression' else numeric_cols[1]))
plt.show()

# ## Phase 5: Feature Engineering
# Create new meaningful features, encode categoricals, and select important features.

df['Price_per_SqFt_Proxy'] = df['Amenities_Count'] / (df['Area_SqFt'] + 1) * 1000
df['Room_Density'] = (df['Bedrooms'] + df['Bathrooms']) / (df['Area_SqFt'] / 1000 + 0.1)
numeric_cols = numeric_cols + ['Price_per_SqFt_Proxy', 'Room_Density']
print("Engineered features added:", ['Price_per_SqFt_Proxy', 'Room_Density'])

X = df[numeric_cols + categorical_cols]
y = df['Price_Lakhs']

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# ## Phase 6: Model Building
# Train at least three ML models and compare.

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42),
}
results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": round(mae,3), "MSE": round(mse,3), "RMSE": round(rmse,3), "R2": round(r2,4)}
    fitted[name] = pipe
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.4f}")

# ## Phase 7: Model Evaluation

results_df = pd.DataFrame(results).T.sort_values("R2", ascending=False)
print(results_df)

plt.figure(figsize=(7,4))
sns.barplot(x=results_df.index, y=results_df["R2"], palette="viridis")
plt.title("Model Comparison — R2 Score")
plt.ylabel("R2 Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

best_name = results_df["R2"].idxmax()
best_pipe = fitted[best_name]
print("Best model:", best_name)

preds = best_pipe.predict(X_test)
plt.figure(figsize=(6,5))
plt.scatter(y_test, preds, alpha=0.4, color="purple")
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
plt.plot(lims, lims, 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Actual vs Predicted — {best_name}")
plt.tight_layout()
plt.show()

# ## Phase 8: Save Best Model for Deployment

import os
os.makedirs("Model", exist_ok=True)
joblib.dump(best_pipe, "Model/house_price_model.pkl")
metadata = {
    "best_model": best_name,
    "task": "regression",
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target": "Price_Lakhs",
    "metrics": results,
}

with open("Model/house_price_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved best_model.pkl and metadata.json")

# ## Conclusion
# The best-performing model was saved and is deployed via a Streamlit application (see `Streamlit_App/app.py`) that accepts user inputs and returns real-time predictions.