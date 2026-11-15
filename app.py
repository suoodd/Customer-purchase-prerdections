import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import date

# ---------------------------------------------------------
# Customer Purchase Prediction - Streamlit Application
# ---------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "customer_purchase_model.pkl"
COLUMNS_PATH = APP_DIR / "model_columns.pkl"

st.set_page_config(
    page_title="Customer Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained preprocessing and Logistic Regression pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Place customer_purchase_model.pkl in the same folder as app.py."
        )

    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {COLUMNS_PATH.name}. "
            "Place model_columns.pkl in the same folder as app.py."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, feature_columns


st.title("Customer Purchase Prediction")
st.write(
    "Enter the customer and sales details below. "
    "The application will use the trained Logistic Regression model "
    "to predict whether the customer is likely to make a purchase."
)

try:
    model, feature_columns = load_model()
except Exception as error:
    st.error(str(error))
    st.stop()

st.info(
    "The prediction is based on the features used during model training. "
    "The model's output is an estimate and should support, not replace, "
    "business decisions."
)

with st.form("purchase_prediction_form"):
    st.subheader("Sales and Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input(
            "Price",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )
        unit_number = st.number_input(
            "Unit Number",
            min_value=0,
            value=1,
            step=1
        )
        start_date = st.date_input(
            "Start Date",
            value=date.today()
        )
        country = st.text_input("Country", placeholder="Enter country code")
        city = st.text_input("City", placeholder="Enter city code")
        product = st.text_input("Product", placeholder="Enter product code")
        competitor = st.text_input(
            "Competitor",
            placeholder="Enter competitor code"
        )

    with col2:
        customer_segment_1 = st.text_input("Customer Segment 1")
        customer_segment_2 = st.text_input("Customer Segment 2")
        customer_segment_3 = st.text_input("Customer Segment 3")
        customer_segment_4 = st.text_input("Customer Segment 4")
        customer_segment_5 = st.text_input("Customer Segment 5")
        activity_1 = st.text_input("Activity 1")
        activity_2 = st.text_input("Activity 2")
        marketing_source = st.text_input("Marketing Source")
        division = st.text_input("Division")

    submitted = st.form_submit_button(
        "Predict Purchase",
        use_container_width=True
    )

if submitted:
    # Create the same engineered date features used during training.
    input_values = {
        "Price": price,
        "Unit Number": unit_number,
        "Start_Year": start_date.year,
        "Start_Month": start_date.month,
        "Start_Quarter": (start_date.month - 1) // 3 + 1,
        "Customer Segment 1": customer_segment_1.strip() or np.nan,
        "Customer Segment 2": customer_segment_2.strip() or np.nan,
        "Customer Segment 3": customer_segment_3.strip() or np.nan,
        "Customer Segment 4": customer_segment_4.strip() or np.nan,
        "Customer Segment 5": customer_segment_5.strip() or np.nan,
        "Activity 1": activity_1.strip() or np.nan,
        "Activity 2": activity_2.strip() or np.nan,
        "Country": country.strip() or np.nan,
        "City": city.strip() or np.nan,
        "Product": product.strip() or np.nan,
        "Competitor": competitor.strip() or np.nan,
        "Marketing Source": marketing_source.strip() or np.nan,
        "Division": division.strip() or np.nan,
    }

    # Match the exact feature order saved with the trained model.
    input_data = pd.DataFrame([input_values])
    input_data = input_data.reindex(columns=feature_columns)

    # The training script converted categorical values to strings.
    categorical_columns = [
        column for column in input_data.columns
        if input_data[column].dtype == "object"
    ]
    for column in categorical_columns:
        input_data[column] = input_data[column].apply(
            lambda value: str(value) if pd.notna(value) else np.nan
        )

    try:
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        purchase_probability = float(probabilities[1])

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted Result: Purchase")
        else:
            st.warning("Predicted Result: No Purchase")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability:.2%}"
        )
        st.progress(min(max(purchase_probability, 0.0), 1.0))

        st.caption(
            "Class 1 means Purchase. Class 0 means No Purchase. "
            "The probability is the model's estimated likelihood of purchase."
        )

    except Exception as error:
        st.error(
            "The prediction could not be completed. "
            f"Please check the entered values and model files. Details: {error}"
        )

st.divider()
st.caption("Customer Purchase Prediction | Machine Learning Project")
