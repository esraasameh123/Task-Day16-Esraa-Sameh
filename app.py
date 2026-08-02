import streamlit as st
import joblib
import pandas as pd


# Load model
model = joblib.load("BaggingClassifier_model.pkl")


st.title("Titanic Survival Prediction 🚢")


# Inputs
Pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

Age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

SibSp = st.number_input(
    "SibSp",
    min_value=0,
    max_value=10,
    value=0
)

Parch = st.number_input(
    "Parch",
    min_value=0,
    max_value=10,
    value=0
)

Fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=10.0
)

Sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

Embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)


if st.button("Predict"):

    # Create dataframe
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Sex": [Sex],
        "Embarked": [Embarked]
    })


    # One Hot Encoding
    input_data = pd.get_dummies(
        input_data,
        columns=["Sex", "Embarked"],
        drop_first=True,
        dtype=int
    )


    # Same columns used in training
    model_columns = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Sex_male",
        "Embarked_Q",
        "Embarked_S"
    ]


    # Match training columns
    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )


    # Prediction
    prediction = model.predict(input_data)


    # Probability
    probability = model.predict_proba(input_data)


    # Result
    if prediction[0] == 1:
        st.success("Passenger Survived ✅")
    else:
        st.error("Passenger Did Not Survive ❌")


    st.write(
        f"Survival Probability: {probability[0][1]*100:.2f}%"
    )