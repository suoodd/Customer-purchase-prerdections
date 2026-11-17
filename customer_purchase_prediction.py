"""
Customer Purchase Prediction Project
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)


# Import Libraries


# Part 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_validate
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)


# Load the Datasets

# Part 2: Load the Datasets

train = pd.read_excel(BASE_DIR / "Training Dataset.xlsx")
test = pd.read_excel(BASE_DIR / "Testing Dateset.xlsx")

print("Training dataset:", train.shape)
print("Testing dataset:", test.shape)

print("\nFirst five rows:")
print(train.head())

print("\nDataset information:")
train.info()


# Data Understanding

print("Statistical Summary:")
print(train.describe(include="all").to_string())

print("\nColumn Names:")
print(train.columns.tolist())

print("\nTarget Distribution:")
print(train["Result"].value_counts())

print("\nTarget Percentages:")
print(train["Result"].value_counts(normalize=True) * 100)

# Check Data Quality

print("Missing Values in Training Data:")
print(train.isnull().sum())

print("\nMissing Values in Testing Data:")
print(test.isnull().sum())

print("\nDuplicate Rows:")
print("Training:", train.duplicated().sum())
print("Testing:", test.duplicated().sum())

print("\nData Types:")
print(train.dtypes)

quality = pd.DataFrame({
    "Column": train.columns,
    "Missing": train.isnull().sum().values,
    "Missing_Percent": (
        train.isnull().mean().values * 100
    ).round(2),
    "Unique_Values": train.nunique().values
})

print("\nData Quality Summary:")
print(quality)

quality.to_csv("data_quality.csv", index=False)


# Data Cleaning


# Part 5: Data Cleaning

train_clean = train.copy()
test_clean = test.copy()

# Remove duplicate records
train_clean = train_clean.drop_duplicates()

# Remove rows with missing target values
train_clean = train_clean.dropna(subset=["Result"])

# Clean text values
for data in [train_clean, test_clean]:

    text_columns = data.select_dtypes(
        include=["object"]
    ).columns

    for col in text_columns:
        data[col] = data[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        data[col] = data[col].replace("", np.nan)

# Convert date columns
for data in [train_clean, test_clean]:

    for col in ["Start Date", "End Date"]:

        if col in data.columns:
            data[col] = pd.to_datetime(
                data[col], errors="coerce"
            )

            data.loc[
                data[col].dt.year <= 1900, col
            ] = pd.NaT

# Check invalid numeric values
for data in [train_clean, test_clean]:

    for col in ["Price", "Unit Number"]:

        if col in data.columns:
            data.loc[data[col] < 0, col] = np.nan

# Check outliers using IQR
for col in ["Price", "Unit Number"]:

    if col in train_clean.columns:

        Q1 = train_clean[col].quantile(0.25)
        Q3 = train_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (
            (train_clean[col] < lower) |
            (train_clean[col] > upper)
        ).sum()

        print(col, "Outliers:", outliers)

# Keep valid outliers because large orders may be real.

print("\nRows Before Cleaning:", len(train))
print("Rows After Cleaning:", len(train_clean))


# Exploratory Data Analysis


# Part 6: Exploratory Data Analysis

# Chart 1: Purchase Distribution

train_clean["Result"].value_counts().sort_index().plot(
    kind="bar", figsize=(6, 4)
)

plt.title("Customer Purchase Distribution")
plt.xlabel("Result (0 = No Purchase, 1 = Purchase)")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("purchase_distribution.png")
plt.show()


# Chart 2: Price Distribution

plt.figure(figsize=(7, 4))

plt.hist(
    train_clean["Price"].dropna(),
    bins=40
)

plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()


# Chart 3: Purchase Rate by Country

country_rate = train_clean.groupby(
    "Country"
)["Result"].agg(["mean", "count"])

country_rate = country_rate[
    country_rate["count"] >= 100
]

country_rate = country_rate.sort_values(
    "mean", ascending=False
).head(10)

country_rate["mean"].plot(
    kind="bar", figsize=(9, 5)
)

plt.title("Purchase Rate by Country")
plt.xlabel("Country")
plt.ylabel("Purchase Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("country_purchase_rate.png")
plt.show()


# Chart 4: Average Price by Purchase Result

train_clean.groupby("Result")["Price"].mean().plot(
    kind="bar", figsize=(6, 4)
)

plt.title("Average Price by Purchase Result")
plt.xlabel("Result")
plt.ylabel("Average Price")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_price.png")
plt.show()


# Chart 5: Monthly Purchase Rate

monthly_rate = train_clean.groupby(
    train_clean["Start Date"].dt.month
)["Result"].mean()

monthly_rate.plot(
    kind="line", marker="o", figsize=(8, 4)
)

plt.title("Purchase Rate by Month")
plt.xlabel("Month")
plt.ylabel("Purchase Rate")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig("monthly_purchase_rate.png")
plt.show()


# Feature Engineering


# Feature Engineering

# Create new features from Start Date
for data in [train_clean, test_clean]:

    data["Start_Year"] = data["Start Date"].dt.year
    data["Start_Month"] = data["Start Date"].dt.month
    data["Start_Quarter"] = data["Start Date"].dt.quarter

# Remove columns not suitable for prediction
drop_columns = [
    "Result",
    "ID",
    "Customer ID",
    "Start Date",
    "End Date",
    "Estimated Win Rate",
    "Manager",
    "Techincal Manager"
]

X = train_clean.drop(
    columns=drop_columns, errors="ignore"
)

y = train_clean["Result"].astype(int)

X_test = test_clean.drop(
    columns=drop_columns, errors="ignore"
)

# Remove columns with one unique value
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

X_test = X_test.drop(
    columns=constant_columns, errors="ignore"
)

# Use the same columns for both datasets
X_test = X_test.reindex(columns=X.columns)

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = [
    col for col in X.columns
    if col not in numeric_columns
]

# Convert categories to text
for col in categorical_columns:

    X[col] = X[col].apply(
        lambda x: str(x) if pd.notna(x) else np.nan
    )

    X_test[col] = X_test[col].apply(
        lambda x: str(x) if pd.notna(x) else np.nan
    )

print("Numeric Columns:", numeric_columns)
print("\nCategorical Columns:", categorical_columns)
print("\nFinal Features:", X.shape[1])


# Split Training and Validation Data


# Split the Data

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Rows:", len(X_train))
print("Validation Rows:", len(X_valid))


# Data Preprocessing


# Data Preprocessing

# Replace missing numeric values with the median
numeric_process = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Replace missing categories and encode them
categorical_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_process, numeric_columns),
    ("cat", categorical_process, categorical_columns)
])


# Build Machine Learning Models


# Build Machine Learning Models

models = {

    "Decision Tree": Pipeline([
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocess", preprocessor),
        ("model", BernoulliNB())
    ])
}

print("Models created:")
for name in models:
    print(name)


# Hyperparameter Tuning


# Hyperparameter Tuning

parameters = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"]
    },

    "Decision Tree": {
        "model__max_depth": [5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 5, 10]
    },

    "Naive Bayes": {
        "model__alpha": [0.1, 0.5, 1, 2]
    }
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

best_models = {}
tuning_results = []

best_models["Baseline"] = models["Baseline"]

for name in parameters:

    print("\nTuning:", name)

    search = RandomizedSearchCV(
        estimator=models[name],
        param_distributions=parameters[name],
        n_iter=5,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_models[name] = search.best_estimator_

    tuning_results.append({
        "Model": name,
        "Best F1": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })

tuning_df = pd.DataFrame(tuning_results)

print("\nTuning Results:")
print(tuning_df.to_string(index=False))

tuning_df.to_csv(
    "hyperparameter_tuning.csv", index=False
)


# Cross-Validation


# Cross-Validation

# Check model performance on the training data

cv_results = []

for name, model in best_models.items():

    print("Checking:", name)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring={
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1": "f1",
            "ROC_AUC": "roc_auc"
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Accuracy": scores["test_Accuracy"].mean(),
        "Precision": scores["test_Precision"].mean(),
        "Recall": scores["test_Recall"].mean(),
        "F1": scores["test_F1"].mean(),
        "ROC_AUC": scores["test_ROC_AUC"].mean()
    })

cv_df = pd.DataFrame(cv_results)

print("\nCross-Validation Results:")
print(cv_df.to_string(index=False))

cv_df.to_csv("cross_validation.csv", index=False)


# Final Model Evaluation


#Final Model Evaluation

results = []

for name, model in best_models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    accuracy = accuracy_score(y_valid, predictions)
    precision = precision_score(
        y_valid, predictions, zero_division=0
    )
    recall = recall_score(
        y_valid, predictions, zero_division=0
    )
    f1 = f1_score(
        y_valid, predictions, zero_division=0
    )
    auc = roc_auc_score(y_valid, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_valid, predictions)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Purchase", "Purchase"]
    ).plot()

    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        name.replace(" ", "_") + "_confusion.png"
    )
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_predictions(
        y_valid, probabilities
    )

    plt.title(name + " ROC Curve")
    plt.tight_layout()
    plt.savefig(
        name.replace(" ", "_") + "_roc.png"
    )
    plt.show()


# Compare Models


# Compare Models

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F1", ascending=False
)

print("Model Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison.csv", index=False
)

results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
].plot(kind="bar", figsize=(11, 6))

plt.title("Machine Learning Model Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# Select the Best Model


# Select the Best Model

# Select the model using training cross-validation F1

candidate_results = cv_df[
    cv_df["Model"] != "Baseline"
]

best_name = candidate_results.sort_values(
    "F1", ascending=False
).iloc[0]["Model"]

print("Selected Model:", best_name)

best_model = best_models[best_name]

# Train the final model using all labelled data
best_model.fit(X, y)

# Save the model
joblib.dump(
    best_model,
    BASE_DIR / "customer_purchase_model.pkl"
)

# Save feature names
joblib.dump(
    X.columns.tolist(),
    BASE_DIR / "model_columns.pkl"
)


# Predict the Testing Dataset


#Predict the Testing Dataset

test_predictions = best_model.predict(X_test)

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

prediction_output = pd.DataFrame({
    "ID": test_clean["ID"],
    "Predicted_Result": test_predictions,
    "Purchase_Probability": test_probabilities
})

print("First 10 Predictions:")
print(prediction_output.head(10))

print("\nPrediction Summary:")
print(prediction_output["Predicted_Result"].value_counts())

prediction_output.to_csv(
    "customer_purchase_predictions.csv",
    index=False
)


# Final Project Summary


# Final Project Summary

print("CUSTOMER PURCHASE PREDICTION PROJECT")
print("-----------------------------------")

print("Training Records:", len(train))
print("Testing Records:", len(test))
print("Predictor Columns:", len(X.columns))

print("\nSelected Model:", best_name)

print("\nModel Performance:")
print(results_df.to_string(index=False))

print("\nProject completed successfully.")


print("\nSaved files are located in:", BASE_DIR)
print("Model file:", BASE_DIR / "customer_purchase_model.pkl")
print("Feature columns:", BASE_DIR / "model_columns.pkl")
