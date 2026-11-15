# Customer Purchase Prediction

## Project Overview

This project uses machine learning to predict whether a customer is likely to make a purchase. The application uses a trained Logistic Regression model and displays the predicted result and estimated purchase probability.

## Models Used

The project compares the following models:

- Logistic Regression
- Decision Tree
- Naive Bayes
- Baseline model

Logistic Regression was selected as the final model based on its F1 score in the validation evaluation.

## Dataset

The project uses two Excel files during model development:

- Training Dataset.xlsx
- Testing Dateset.xlsx

The training dataset contains 177,086 records and the testing dataset contains 44,272 records. The testing dataset does not contain the actual target labels, so its predictions are not used to calculate model accuracy.

The target column is `Result`:

- `0` = No Purchase
- `1` = Purchase

The Excel datasets are not required to run the deployed Streamlit application because the trained model is saved in the model file.

## Model Performance

The final validation results reported by the Python script are:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 65.73% | 58.83% | 65.89% | 62.16% | 71.89% |
| Decision Tree | 69.33% | 68.74% | 51.75% | 59.05% | 74.43% |
| Naive Bayes | 63.87% | 57.28% | 60.67% | 58.93% | 68.35% |
| Baseline | 57.27% | 0.00% | 0.00% | 0.00% | 50.00% |

Logistic Regression was selected because it achieved the highest F1 score among the evaluated models. The Decision Tree achieved higher accuracy and ROC-AUC, but its F1 score was lower.

## Project Files

Keep these files in the same project folder:

- `app.py` — Streamlit application.
- `customer_purchase_model.pkl` — saved trained model pipeline.
- `model_columns.pkl` — feature columns used by the model.
- `requirements.txt` — Python libraries required by the application.

The training script and analysis outputs can also be included in the repository if required by the project submission.

## Run the Application Locally

### 1. Install the required libraries

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser. If it does not open automatically, visit:

`http://localhost:8501`

## How to Use the Application

1. Enter the customer and sales details in the form.
2. Click **Predict Purchase**.
3. The application displays either Purchase or No Purchase.
4. The purchase probability shows the model's estimated likelihood of a purchase.

## Deployment

The application can be deployed through Streamlit Community Cloud by connecting this GitHub repository and selecting `app.py` as the main file.

After deployment, add the public Streamlit application link to the project report.

## Important Note

The prediction is an estimate produced by the trained model. It should be used as supporting information and not as the only basis for business decisions.
