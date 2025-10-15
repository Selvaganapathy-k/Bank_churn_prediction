import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).parent / "Churn_Modelling.csv"
MODEL_PATH = Path(__file__).parent / "churn_model.pkl"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    y = df["Exited"]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Apply power transform to numeric features first to reduce skew where appropriate
    numeric_transformer = Pipeline(steps=[("power", PowerTransformer(method="yeo-johnson")),
                                          ("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="passthrough")

    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("classifier", DecisionTreeClassifier(random_state=42))])
    return model


def train_and_save(df: pd.DataFrame, model_path: Path) -> Pipeline:
    model = build_pipeline(df)
    X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
    y = df["Exited"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    # Save model and also evaluation on training/test
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc, y_test, y_pred


def load_model(model_path: Path) -> Pipeline | None:
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            # Common causes: model was pickled with a different sklearn version
            # or the file is corrupted. Don't let the app crash — move the file
            # aside and return None so a new model can be trained.
            try:
                backup = model_path.with_suffix(model_path.suffix + ".corrupt")
                model_path.rename(backup)
            except Exception:
                backup = None
            # Use Streamlit to inform the user in the UI; logs will contain full traceback
            try:
                st.warning(
                    "Saved model could not be loaded (incompatible/corrupt). It was moved aside and a new model can be trained."
                )
                if backup is not None:
                    st.info(f"Old model moved to: {backup.name}")
            except Exception:
                # If Streamlit isn't initialized (rare), ignore UI calls and continue
                pass
            return None
    return None


def predict_single(model: Pipeline, input_df: pd.DataFrame) -> np.ndarray:
    return model.predict(input_df)


def main():
    st.set_page_config(page_title="Bank Churn - Demo", layout="centered")
    st.title("Bank Customer Churn - Demo App")

    if not DATA_PATH.exists():
        st.error(f"Data file not found at {DATA_PATH}. Place `Churn_Modelling.csv` next to this app.")
        return

    df = load_data(DATA_PATH)

    st.sidebar.header("Actions")
    action = st.sidebar.radio("Choose action", ["Overview", "Train / Load Model", "Predict single"])

    if action == "Overview":
        st.header("Dataset overview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write(df.describe())

    elif action == "Train / Load Model":
        st.header("Train or load model")
        # Allow the user to remove a saved model from the UI (helpful if the pickle
        # becomes incompatible or corrupt in the deployed environment).
        if st.button("Reset saved model"):
            try:
                if MODEL_PATH.exists():
                    MODEL_PATH.unlink()
                    st.success("Saved model file removed. You can train a fresh model now.")
                else:
                    st.info("No saved model file found to remove.")
            except Exception as e:
                st.error(f"Could not remove saved model: {e}")
        model = load_model(MODEL_PATH)
        if model is None:
            st.info("No saved model found — training a new model. This may take a few seconds.")
            with st.spinner("Training model..."):
                model, acc, y_test, y_pred = train_and_save(df, MODEL_PATH)
            st.success(f"Training complete. Test accuracy: {acc:.4f}")
        else:
            st.success("Loaded saved model from disk.")

        # If we have a saved model, show a small evaluation using a fresh split
        model = load_model(MODEL_PATH)
        if model is not None:
            X = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
            y = df["Exited"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write("Test accuracy:", acc)
            st.write("Classification report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion matrix:")
            st.write(confusion_matrix(y_test, y_pred))

    elif action == "Predict single":
        st.header("Predict churn for a single customer")
        st.write("Provide the customer's information and click Predict.")

        # Prepare default values from dataset median / mode
        sample = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"]).iloc[0]

        with st.form("predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                credit_score = st.number_input("CreditScore", value=int(df["CreditScore"].median()))
                geography = st.selectbox("Geography", options=df["Geography"].unique().tolist())
                gender = st.selectbox("Gender", options=df["Gender"].unique().tolist())
                age = st.number_input("Age", value=int(df["Age"].median()))
                tenure = st.number_input("Tenure", value=int(df["Tenure"].median()))
            with col2:
                balance = st.number_input("Balance", value=float(df["Balance"].median()))
                num_products = st.number_input("NumOfProducts", value=int(df["NumOfProducts"].median()))
                has_cr_card = st.selectbox("HasCrCard", options=[0, 1], index=1)
                is_active = st.selectbox("IsActiveMember", options=[0, 1], index=1)
                estimated_salary = st.number_input("EstimatedSalary", value=float(df["EstimatedSalary"].median()))

            submitted = st.form_submit_button("Predict")

        if submitted:
            model = load_model(MODEL_PATH)
            if model is None:
                st.warning("No model found — training one now.")
                with st.spinner("Training model..."):
                    model, _, _, _ = train_and_save(df, MODEL_PATH)

            input_df = pd.DataFrame([{
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary,
            }])

            pred = predict_single(model, input_df)[0]
            st.write("Prediction (1 = churn, 0 = no churn):", int(pred))


if __name__ == "__main__":
    main()
