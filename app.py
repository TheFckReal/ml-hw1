import pickle
import re
from pathlib import Path
from tempfile import TemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phik  # noqa: F401 - needed for .phik_matrix() accessor
import seaborn as sns
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


DATA_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
MODEL_PATH = Path("./models/ridge_log.pkl")
PIPELINE_PATH = Path("./models/data_pipeline.pkl")

NUMERIC_COLS_TO_CLEAN = ["mileage", "engine", "max_power"]
NUMERIC_COLS = ["mileage", "engine", "max_power", "torque", "seats", "km_driven", "max_torque_rpm"]
KGM_TO_NM = 9.80665

st.set_page_config(page_title="Car Price Prediction", layout="wide")


class FeatureEnrichment(BaseEstimator, TransformerMixin):
    """
    Feature enrichment transformer.
    Adds the following features:
    - age: the age of the car
    - power_per_liter: the power of the car per liter of engine volume
    - km_per_year: the kilometers driven per year
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if "year" in X_copy.columns:
            current_year = getattr(self, "max_year", X_copy["year"].max())
            X_copy["age"] = current_year - X_copy["year"]
        if "engine" in X_copy.columns and "max_power" in X_copy.columns:
            X_copy["power_per_liter"] = X_copy["max_power"] / (X_copy["engine"] + 1e-5)
        if "km_driven" in X_copy.columns and "age" in X_copy.columns:
            X_copy["km_per_year"] = X_copy["km_driven"] / (X_copy["age"] + 1)
        return X_copy


# --- Data Preprocessing ---
def _parse_torque_value(x: str) -> tuple[float, float]:
    """Parses the torque string and returns (torque_nm, rpm)."""
    if pd.isna(x):
        return np.nan, np.nan

    x_clean = str(x).lower().replace(",", "")
    multiplier = KGM_TO_NM if "kgm" in x_clean else 1.0

    x_clean = x_clean.replace("at", "@").replace("/", "@")
    parts = x_clean.split("@")

    torque_val = rpm_val = np.nan

    if parts:
        match = re.search(r"([\d.]+)", parts[0])
        if match:
            torque_val = float(match.group(1)) * multiplier

    if len(parts) > 1:
        matches = re.findall(r"([\d.]+)", " ".join(parts[1:]))
        if matches:
            rpm_val = float(matches[0])

    return torque_val, rpm_val


def preprocess_dataframe(
    df: pd.DataFrame, fill_values: pd.Series | None = None
) -> pd.DataFrame:
    """General preprocessing of the dataframe."""
    df = df.copy()

    # Clean measurement units
    for col in NUMERIC_COLS_TO_CLEAN:
        if col in df.columns:
            df[col] = df[col].astype(str).str.split(expand=True)[0]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse torque
    if "torque" in df.columns:
        parsed = df["torque"].apply(lambda x: pd.Series(_parse_torque_value(x)))
        df["torque"], df["max_torque_rpm"] = parsed[0], parsed[1]

    # Extract brand from name
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.split(n=1).str[0]

    # Fill missing values
    if fill_values is not None:
        for col in fill_values.index:
            if col in df.columns:
                df[col] = df[col].fillna(fill_values[col])

    # Ensure numeric types
    for col in ["engine", "seats", "km_driven", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def predict_price(model, pipeline, df: pd.DataFrame) -> np.ndarray:
    """Performs price prediction."""
    X = pipeline.transform(df)
    log_pred = np.clip(model.predict(X), -50, 50)
    return np.exp(log_pred)


# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"Could not load the data: {e}")
        return None


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists() or not PIPELINE_PATH.exists():
        st.error(f"Model or pipeline file not found: {MODEL_PATH}, {PIPELINE_PATH}")
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(PIPELINE_PATH, "rb") as f:
            pipeline = pickle.load(f)
        return model, pipeline
    except Exception as e:
        st.error(f"Load models error: {e}")
        return None, None


@st.cache_data
def get_fill_values(df_train: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Calculates medians for filling missing values."""
    df_clean = preprocess_dataframe(df_train)
    cols = [c for c in NUMERIC_COLS if c in df_clean.columns]
    return df_clean[cols].median(), df_clean


# --- Pages ---
def show_eda(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["selling_price"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Pair distribution plot")
    pair_plot = sns.pairplot(
        df.select_dtypes(include=["number"]), diag_kind="kde", corner=True
    )
    st.pyplot(pair_plot.figure)

    st.subheader("Phik correlation matrix")
    try:
        phik_matrix = df.phik_matrix()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Phik is now available ({e}). Spearman correlation is shown.")
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                numeric_df.corr("spearman", numeric_only=True),
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                ax=ax,
            )
            st.pyplot(fig)


def show_prediction(model, pipeline, fill_values: pd.Series):
    st.header("Price Prediction")

    tab1, tab2 = st.tabs(["Single Item", "Batch Upload"])

    with tab1:
        _show_single_prediction(model, pipeline, fill_values)

    with tab2:
        _show_batch_prediction(model, pipeline, fill_values)


def _show_single_prediction(model, pipeline, fill_values: pd.Series):
    st.subheader("Enter Car Details")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Car Name", "Maruti Swift")
        year = st.number_input("Year", 1990, 2025, 2018)
        km_driven = st.number_input("Kilometers Driven", 0, 1_000_000, 50_000)
        fuel = st.selectbox("Fuel", ["Diesel", "Petrol", "CNG", "LPG"])
        seller_type = st.selectbox(
            "Seller Type", ["Individual", "Dealer", "Trustmark Dealer"]
        )

    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox(
            "Owner",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car",
            ],
        )
        mileage = st.text_input("Mileage (e.g. 23.4 kmpl)", "20.0 kmpl")
        engine = st.text_input("Engine (e.g. 1248 CC)", "1200 CC")
        max_power = st.text_input("Max Power (e.g. 74 bhp)", "75 bhp")
        torque = st.text_input("Torque (e.g. 190Nm@ 2000rpm)", "190Nm@ 2000rpm")
        seats = st.number_input("Seats", 2, 14, 5)

    if st.button("Predict", key="single"):
        input_df = pd.DataFrame(
            [
                {
                    "name": name,
                    "year": year,
                    "km_driven": km_driven,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "torque": torque,
                    "seats": seats,
                }
            ]
        )

        try:
            processed = preprocess_dataframe(input_df, fill_values)
            price = predict_price(model, pipeline, processed)[0]
            st.success(f"Predicted Selling Price: {price:,.2f}")
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")


def _show_batch_prediction(model, pipeline, fill_values: pd.Series):
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is None:
        return

    input_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", input_df.head())

    if st.button("Predict Batch", key="batch"):
        try:
            processed = preprocess_dataframe(input_df.copy(), fill_values)
            processed = processed.drop(columns=["selling_price"], errors="ignore")

            prices = predict_price(model, pipeline, processed)
            input_df["predicted_price"] = prices

            st.write("Predictions:", input_df.head())

            with TemporaryFile("w+b") as f:
                input_df.to_csv(f, index=False)
                f.seek(0)
                st.download_button(
                    "Download Results", f.read(), "predictions.csv", "text/csv"
                )
        except Exception as e:
            st.error(f"Error: {e}")


def show_weights(model, pipeline, df_clean: pd.DataFrame):
    st.header("Model Weights Visualization")
    num_features = st.number_input("Number of top features to show", value=20, min_value=1, max_value=100)
    try:
        df_sample = df_clean.head(5).copy()
        df_sample["name"] = df_sample["name"].astype(str).str.split(n=1).str[0]

        for col in NUMERIC_COLS:
            if col in df_sample.columns:
                median = df_sample[col].median()
                df_sample[col] = df_sample[col].fillna(
                    median if pd.notna(median) else 0
                )  # type: ignore

        X_sample = df_sample.drop(columns=["selling_price"], errors="ignore")
        X_enriched = pipeline.named_steps["fe"].transform(X_sample)
        prep = pipeline.named_steps["prep"]

        feature_names = (
            prep.get_feature_names_out(X_enriched.columns)
            if hasattr(prep, "get_feature_names_out")
            else [f"Feature_{i}" for i in range(len(model.coef_))]
        )

        feat_df = pd.DataFrame(
            {
                "Feature": feature_names[: len(model.coef_)],
                "Weight": model.coef_,
                "abs_weight": np.abs(model.coef_),
            }
        ).nlargest(num_features, "abs_weight")

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.barplot(
            data=feat_df,
            x="Weight",
            y="Feature",
            ax=ax,
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        st.subheader(f"Top {num_features} Features by Weight")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Не удалось извлечь имена признаков: {e}")
        st.write("Model Coefs:", model.coef_)


# --- Main ---
def main():
    st.title("🚗 Auto Price Prediction Dashboard")

    df_train = load_data()
    model, pipeline = load_models()

    if df_train is None or model is None or pipeline is None:
        st.warning("The application cannot work without data and models.")
        return

    fill_values, df_clean = get_fill_values(df_train)

    if "fe" in pipeline.named_steps:
        pipeline.named_steps["fe"].max_year = df_train["year"].max()

    pages = {
        "EDA": lambda: show_eda(preprocess_dataframe(df_train)),
        "Prediction": lambda: show_prediction(model, pipeline, fill_values),
        "Model Weights": lambda: show_weights(model, pipeline, df_clean),
    }

    page = st.sidebar.selectbox("Choose a page", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
