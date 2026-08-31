import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main {
        padding-top: 1rem;
    }

    .forecast-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .forecast-subtitle {
        font-size: 18px;
        color: #888;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #111827;
        border: 1px solid #374151;
        text-align: center;
    }

    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        font-size: 17px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_FILE = "xgboost_units_sold_model.joblib"
FEATURE_FILE = "xgboost_features.pkl"

MODEL_RMSE = 7.793
MODEL_MAE = 5.806
MODEL_WAPE = 15.208
MODEL_ACCURACY = 84.792


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    return model


@st.cache_data
def load_features():

    try:
        with open(FEATURE_FILE, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):

            if "features" in obj:
                return obj["features"]

            if "feature_names" in obj:
                return obj["feature_names"]

        if isinstance(obj, list):
            return obj

    except Exception:
        pass

    # Fallback: your final TOP_12 feature set
    return [
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


try:

    model = load_model()
    feature_names = load_features()

except Exception as e:

    st.error("Unable to load the trained XGBoost model.")
    st.code(str(e))
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="forecast-title">📈 Sales Forecasting System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="forecast-subtitle">'
    'XGBoost-powered future sales forecasting'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Forecast Parameters")

st.sidebar.markdown("### Product Information")

product = st.sidebar.selectbox(
    "Product",
    [
        "P001",
        "P002",
        "P003",
        "P004",
        "P005",
        "P006",
        "P007",
        "P008"
    ]
)

store = st.sidebar.selectbox(
    "Store",
    [
        "S01",
        "S02",
        "S03",
        "S04"
    ]
)

category = st.sidebar.selectbox(
    "Category",
    [
        "Electronics",
        "Home",
        "Clothing",
        "Grocery",
        "Beauty"
    ]
)

st.sidebar.markdown("### Sales Conditions")

price = st.sidebar.number_input(
    "💰 Price",
    min_value=0.0,
    value=1500.0,
    step=10.0
)

discount = st.sidebar.number_input(
    "🏷️ Discount %",
    min_value=0.0,
    max_value=100.0,
    value=10.0,
    step=1.0
)

promotion = st.sidebar.selectbox(
    "📣 Promotion",
    ["No", "Yes"]
)

stock = st.sidebar.selectbox(
    "📦 Stock Availability",
    ["Available", "Out of Stock"]
)

holiday = st.sidebar.selectbox(
    "🎉 Holiday",
    ["No", "Yes"]
)

local_event = st.sidebar.selectbox(
    "📍 Local Event",
    ["No", "Yes"]
)

weekend = st.sidebar.selectbox(
    "📅 Weekend",
    ["No", "Yes"]
)

competitor_price = st.sidebar.number_input(
    "🏪 Competitor Price",
    min_value=0.0,
    value=1700.0,
    step=10.0
)

economic_indicator = st.sidebar.number_input(
    "📊 Economic Indicator",
    value=1.0,
    step=0.1
)

marketing_spend = st.sidebar.number_input(
    "📢 Marketing Spend",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

st.sidebar.markdown("### Historical Sales")

sales_3_periods_ago = st.sidebar.number_input(
    "Sales 3 Periods Ago",
    min_value=0.0,
    value=40.0,
    step=1.0
)

current_reference_sales = st.sidebar.number_input(
    "Current Reference Sales",
    min_value=0.0,
    value=40.0,
    step=1.0
)

st.sidebar.markdown("### Forecast Horizon")

forecast_days = st.sidebar.slider(
    "Forecast Horizon",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(
    price,
    competitor_price,
    discount,
    promotion,
    stock,
    holiday,
    local_event,
    weekend,
    economic_indicator,
    marketing_spend,
    sales_3_periods_ago,
    current_reference_sales
):

    price_gap = price - competitor_price

    if competitor_price != 0:
        price_ratio = price / competitor_price
    else:
        price_ratio = 1.0

    promotion_flag = 1 if promotion == "Yes" else 0
    stock_availability = 1 if stock == "Available" else 0
    holiday_flag = 1 if holiday == "Yes" else 0
    local_event_flag = 1 if local_event == "Yes" else 0
    is_weekend = 1 if weekend == "Yes" else 0

    # Historical change feature.
    #
    # IMPORTANT:
    # The deployed interface uses the supplied historical reference
    # values instead of inventing target values.
    sales_change_3 = (
        current_reference_sales - sales_3_periods_ago
    )

    row = {
        "Stock_Availability": stock_availability,
        "Promotion_Flag": promotion_flag,
        "Discount_Percentage": discount,
        "Local_Event_Flag": local_event_flag,
        "Competitor_Price": competitor_price,
        "Price": price,
        "Marketing_Spend": marketing_spend,
        "price_gap": price_gap,
        "Economic_Indicator": economic_indicator,
        "price_ratio": price_ratio,
        "Is_Weekend": is_weekend,
        "sales_change_3": sales_change_3
    }

    # Make sure feature order exactly matches training
    X = pd.DataFrame([row])

    X = X.reindex(columns=feature_names, fill_value=0)

    return X


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sales(X):

    prediction = model.predict(X)

    prediction = float(prediction[0])

    # Units sold cannot be negative
    prediction = max(0, prediction)

    return prediction


# ============================================================
# FORECAST BUTTON
# ============================================================

st.sidebar.markdown("---")

generate = st.sidebar.button(
    "🔮 Generate Forecast",
    type="primary"
)


# ============================================================
# FORECAST
# ============================================================

if generate:

    predictions = []

    start_date = datetime.today().date()

    for i in range(forecast_days):

        forecast_date = start_date + timedelta(days=i)

        # Weekend changes by forecast date
        is_forecast_weekend = (
            1 if forecast_date.weekday() >= 5 else 0
        )

        # Generate model input
        X = create_features(
            price=price,
            competitor_price=competitor_price,
            discount=discount,
            promotion=promotion,
            stock=stock,
            holiday=holiday,
            local_event=local_event,
            weekend="Yes" if is_forecast_weekend else "No",
            economic_indicator=economic_indicator,
            marketing_spend=marketing_spend,
            sales_3_periods_ago=sales_3_periods_ago,
            current_reference_sales=current_reference_sales
        )

        prediction = predict_sales(X)

        predictions.append({
            "Forecast Date": forecast_date,
            "Predicted Units Sold": round(prediction)
        })

    forecast_df = pd.DataFrame(predictions)

    total_forecast = int(
        forecast_df["Predicted Units Sold"].sum()
    )

    average_forecast = round(
        forecast_df["Predicted Units Sold"].mean()
    )

    highest_forecast = int(
        forecast_df["Predicted Units Sold"].max()
    )

    lowest_forecast = int(
        forecast_df["Predicted Units Sold"].min()
    )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    st.success(
        f"Forecast generated successfully for "
        f"{product} at {store}."
    )


    # ========================================================
    # FORECAST SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Forecast Summary</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Forecast",
            f"{total_forecast:,} Units"
        )

    with c2:
        st.metric(
            "Average Daily Sales",
            f"{average_forecast} Units"
        )

    with c3:
        st.metric(
            "Highest Daily Sales",
            f"{highest_forecast} Units"
        )

    with c4:
        st.metric(
            "Lowest Daily Sales",
            f"{lowest_forecast} Units"
        )


    # ========================================================
    # FORECAST CHART
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Sales Forecast</div>',
        unsafe_allow_html=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Forecast Date"],
            y=forecast_df["Predicted Units Sold"],
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3)
        )
    )

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Predicted Units Sold",
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # FORECAST TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Future Sales Forecast</div>',
        unsafe_allow_html=True
    )

    display_df = forecast_df.copy()

    display_df["Forecast Date"] = pd.to_datetime(
        display_df["Forecast Date"]
    ).dt.strftime("%d-%m-%Y")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown("---")

    with st.expander("📌 Model Performance"):

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "RMSE",
            f"{MODEL_RMSE:.3f}"
        )

        m2.metric(
            "MAE",
            f"{MODEL_MAE:.3f}"
        )

        m3.metric(
            "WAPE",
            f"{MODEL_WAPE:.3f}%"
        )

        m4.metric(
            "Accuracy",
            f"{MODEL_ACCURACY:.3f}%"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("🔎 Forecast Input Parameters"):

        input_summary = pd.DataFrame({
            "Parameter": [
                "Product",
                "Store",
                "Category",
                "Price",
                "Discount",
                "Promotion",
                "Stock Availability",
                "Holiday",
                "Local Event",
                "Weekend",
                "Competitor Price",
                "Economic Indicator",
                "Marketing Spend",
                "Sales 3 Periods Ago",
                "Current Reference Sales"
            ],

            "Value": [
                product,
                store,
                category,
                price,
                discount,
                promotion,
                stock,
                holiday,
                local_event,
                weekend,
                competitor_price,
                economic_indicator,
                marketing_spend,
                sales_3_periods_ago,
                current_reference_sales
            ]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )


else:

    # ========================================================
    # LANDING PAGE
    # ========================================================

    st.info(
        "Select the forecast parameters from the sidebar "
        "and click **Generate Forecast**."
    )

    st.markdown(
        '<div class="section-title">🎯 Model Performance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "RMSE",
            f"{MODEL_RMSE:.3f}"
        )

    with c2:
        st.metric(
            "MAE",
            f"{MODEL_MAE:.3f}"
        )

    with c3:
        st.metric(
            "WAPE",
            f"{MODEL_WAPE:.3f}%"
        )

    with c4:
        st.metric(
            "Accuracy",
            f"{MODEL_ACCURACY:.3f}%"
        )

    st.markdown("---")

    st.markdown(
        """
        ### How the forecasting system works

        **1.** Select the product and store.

        **2.** Enter the current business conditions.

        **3.** Choose how many future days to forecast.

        **4.** Click **Generate Forecast**.

        **5.** The XGBoost model generates the expected units sold.

        **6.** Review the forecast graph and daily forecast table.

        **7.** Download the forecast as CSV.
        """
    )
