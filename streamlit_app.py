import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting ML",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("xgboost_units_sold_model.joblib")
    return model


@st.cache_resource
def load_features():
    with open("xgboost_features.pkl", "rb") as f:
        features = pickle.load(f)
    return features


try:
    model = load_model()
    saved_features = load_features()

except Exception as e:
    st.error(f"Unable to load the model files: {e}")
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📈 E-Commerce Sales Forecasting")
st.write(
    "XGBoost-based Units Sold Prediction"
)

st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("RMSE", "7.793")

with col2:
    st.metric("MAE", "5.806")

with col3:
    st.metric("WAPE", "15.208%")

with col4:
    st.metric("Accuracy", "84.792%")


st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter Sales Information")

col1, col2, col3 = st.columns(3)

with col1:

    price = st.number_input(
        "Price",
        min_value=0.0,
        value=2500.0,
        step=10.0
    )

    competitor_price = st.number_input(
        "Competitor Price",
        min_value=0.0,
        value=2500.0,
        step=10.0
    )

    discount = st.number_input(
        "Discount Percentage",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0
    )

    marketing_spend = st.number_input(
        "Marketing Spend",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )


with col2:

    stock = st.selectbox(
        "Stock Availability",
        options=[0, 1],
        format_func=lambda x:
            "Available" if x == 1 else "Out of Stock"
    )

    promotion = st.selectbox(
        "Promotion",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    local_event = st.selectbox(
        "Local Event",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )

    weekend = st.selectbox(
        "Weekend",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


with col3:

    economic_indicator = st.number_input(
        "Economic Indicator",
        value=100.0,
        step=0.1
    )

    sales_lag_3 = st.number_input(
        "Sales 3 Periods Ago",
        min_value=0.0,
        value=40.0,
        step=1.0
    )

    sales_current_reference = st.number_input(
        "Current Reference Sales",
        min_value=0.0,
        value=40.0,
        step=1.0
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

price_gap = price - competitor_price

if competitor_price != 0:
    price_ratio = price / competitor_price
else:
    price_ratio = 1.0


sales_change_3 = sales_current_reference - sales_lag_3


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button(
    "🔮 Predict Units Sold",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE FEATURE DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Stock_Availability": [stock],
        "Promotion_Flag": [promotion],
        "Discount_Percentage": [discount],
        "Local_Event_Flag": [local_event],
        "Competitor_Price": [competitor_price],
        "Price": [price],
        "Marketing_Spend": [marketing_spend],
        "price_gap": [price_gap],
        "Economic_Indicator": [economic_indicator],
        "price_ratio": [price_ratio],
        "Is_Weekend": [weekend],
        "sales_change_3": [sales_change_3]
    })


    # --------------------------------------------------------
    # ENSURE CORRECT FEATURE ORDER
    # --------------------------------------------------------

    required_features = [
        "Stock_Availability",
        "Promotion_Flag",
        "Discount_Percentage",
        "Local_Event_Flag",
        "Competitor_Price",
        "Price",
        "Marketing_Spend",
        "price_gap",
        "Economic_Indicator",
        "price_ratio",
        "Is_Weekend",
        "sales_change_3"
    ]


    missing_features = [
        f for f in required_features
        if f not in input_data.columns
    ]

    if missing_features:

        st.error(
            f"Missing features: {missing_features}"
        )

        st.stop()


    input_data = input_data[required_features]


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        prediction = max(0, prediction)

        prediction = round(prediction)


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully!")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Units Sold",
                f"{prediction} units"
            )

        with result_col2:

            st.metric(
                "Price Gap",
                f"{price_gap:,.2f}"
            )

        with result_col3:

            st.metric(
                "Price Ratio",
                f"{price_ratio:.2f}"
            )


        # ----------------------------------------------------
        # FEATURE SUMMARY
        # ----------------------------------------------------

        st.subheader("Prediction Inputs")

        display_data = input_data.copy()

        display_data = display_data.T

        display_data.columns = ["Value"]

        st.dataframe(
            display_data,
            use_container_width=True
        )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Information")

st.write(
    """
    **Model:** XGBoost Regressor

    **Target:** Units Sold

    **Test RMSE:** 7.793

    **Test MAE:** 5.806

    **Test WAPE:** 15.208%

    **Test Accuracy:** 84.792%
    """
)
