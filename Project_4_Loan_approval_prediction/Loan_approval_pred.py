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

df = pd.read_csv("loan_approval_dataset.csv")
print("Shape:", df.shape)
df.head()

df.info()

df.describe(include='all').T

# ## Phase 3: Data Preprocessing
# Handle missing values, remove duplicates, treat outliers, encode categorical features.

print("Missing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

dup_count = df.duplicated(subset=[c for c in df.columns if c != 'Loan_ID']).sum()
print("\nDuplicate rows:", dup_count)
df = df.drop_duplicates(subset=[c for c in df.columns if c != 'Loan_ID']).reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

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

plt.figure(figsize=(5,4))
sns.countplot(x=df['Loan_Status'], palette="Set2")
plt.title("Target Class Balance: Loan_Status")
plt.show()
print(df['Loan_Status'].value_counts(normalize=True))

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
corr = df[numeric_cols + (["Loan_Status"] if 'classification'=='regression' else [])].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,5))
plt.scatter(df[numeric_cols[0]], df['Loan_Status'] if 'classification'=='regression' else df[numeric_cols[1]], alpha=0.4, color="teal")
plt.xlabel(numeric_cols[0])
plt.ylabel('Loan_Status' if 'classification'=='regression' else numeric_cols[1])
plt.title(f"Scatter: {numeric_cols[0]} vs " + ('Loan_Status' if 'classification'=='regression' else numeric_cols[1]))
plt.show()

# ## Phase 5: Feature Engineering
# Create new meaningful features, encode categoricals, and select important features.

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['Loan_Income_Ratio'] = df['LoanAmount'] / (df['Total_Income'] / 1000 + 1)
numeric_cols = numeric_cols + ['Total_Income', 'Loan_Income_Ratio']
print("Engineered features added:", ['Total_Income', 'Loan_Income_Ratio'])

X = df[numeric_cols + categorical_cols]
y = df['Loan_Status']

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

# ## Phase 6: Model Building
# Train at least three ML models and compare.

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=250, max_depth=8, class_weight="balanced", random_state=42),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=42),
}
results = {}
fitted = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results[name] = {"Accuracy": round(acc,4), "Precision": round(prec,4), "Recall": round(rec,4),
                      "F1": round(f1,4), "ROC_AUC": round(auc,4)}
    fitted[name] = pipe
    print(f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

# ## Phase 7: Model Evaluation

results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
print(results_df)

plt.figure(figsize=(7,4))
sns.barplot(x=results_df.index, y=results_df["F1"], palette="mako")
plt.title("Model Comparison — F1 Score")
plt.ylabel("F1 Score")
plt.xlabel("Model")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

best_name = results_df["F1"].idxmax()
best_pipe = fitted[best_name]
print("Best model:", best_name)

preds = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.show()

# ## Phase 8: Save Best Model for Deployment

import os
os.makedirs("Model", exist_ok=True)
joblib.dump(best_pipe, "Model/loan_approval_model.pkl")
metadata = {
    "best_model": best_name,
    "task": "classification",
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target": "Loan_Status",
    "metrics": results,
}

metadata["classes"] = list(le.classes_)

with open("Model/loan_approval_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved best_model.pkl and metadata.json")

# ## Conclusion
# The best-performing model was saved and is deployed via a Streamlit application (see `Streamlit_App/app.py`) that accepts user inputs and returns real-time predictions.