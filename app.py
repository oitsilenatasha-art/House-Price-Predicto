import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 HomeValue AI")
st.subheader("House Price Prediction Demo App")

st.write(
    """
    This app estimates a house sale price using a machine learning model trained
    on housing data. It is a learning prototype and should not replace a professional valuation.
    """
)

MODEL_PATH = "house_price_model_pipeline.pkl"
METADATA_PATH = "house_price_app_metadata.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


try:
    model, metadata = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run the notebook first to generate "
        "`house_price_model_pipeline.pkl` and `house_price_app_metadata.pkl`, "
        "then place them in the same folder as this app."
    )
    st.stop()


numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
categorical_options = metadata["categorical_options"]
numeric_defaults = metadata["numeric_defaults"]


st.sidebar.header("Enter House Details")

user_input = {}


for feature in numeric_features:
    default_value = numeric_defaults.get(feature, 0)

    if "Year" in feature:
        value = st.sidebar.number_input(
            feature,
            min_value=1800,
            max_value=2030,
            value=int(default_value),
            step=1
        )

    elif feature == "OverallQual":
        value = st.sidebar.slider(
            feature,
            min_value=1,
            max_value=10,
            value=int(round(default_value))
        )

    else:
        value = st.sidebar.number_input(
            feature,
            min_value=0.0,
            value=float(default_value),
            step=100.0
        )

    user_input[feature] = value


for feature in categorical_features:
    options = categorical_options.get(feature, [])

    if not options:
        options = ["Unknown"]

    value = st.sidebar.selectbox(
        feature,
        options=options
    )

    user_input[feature] = value


input_df = pd.DataFrame([user_input])

st.write("### Input Summary")
st.dataframe(input_df)


if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Estimated Sale Price: ${prediction:,.2f}")

        lower_bound = prediction * 0.90
        upper_bound = prediction * 1.10

        st.info(
            f"Suggested interpretation range: "
            f"${lower_bound:,.2f} to ${upper_bound:,.2f}. "
            "This range is only a simple uncertainty guide for demo purposes."
        )

        st.write("### Business Interpretation")

        st.write(
            """
            The model uses property characteristics such as quality, size, garage
            information, age, and location-related features to estimate a likely
            sale price.

            A real business should validate the model using current local market
            data before using it.
            """
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.write("---")

st.caption(
    "HomeValue AI is a teaching demo for Data Science, Machine Learning, "
    "and Streamlit deployment."
)
