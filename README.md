# Titanic Survival Prediction 🚢

## Project Overview

This project is a Machine Learning classification project that predicts whether a passenger survived the Titanic disaster or not.

The goal is to build a complete ML workflow starting from data preprocessing, model training, evaluation, saving the best model, and deploying it using a Streamlit web application.

---

## Problem Statement

The Titanic dataset contains information about passengers such as age, gender, passenger class, fare, and family members.

The objective is to train a classification model that can predict the survival status of passengers:

* **0 → Did Not Survive**
* **1 → Survived**

---

## Dataset

The project uses the Titanic dataset containing passenger information.

### Features

| Feature     | Description                          |
| ----------- | ------------------------------------ |
| PassengerId | Unique identifier for each passenger |
| Pclass      | Passenger class (1st, 2nd, 3rd)      |
| Name        | Passenger name                       |
| Sex         | Passenger gender                     |
| Age         | Passenger age                        |
| SibSp       | Number of siblings/spouses aboard    |
| Parch       | Number of parents/children aboard    |
| Ticket      | Ticket number                        |
| Fare        | Ticket price                         |
| Cabin       | Cabin number                         |
| Embarked    | Port of embarkation                  |

### Target Variable

| Column   | Description              |
| -------- | ------------------------ |
| Survived | Survival status (0 or 1) |

---

# Data Preprocessing

The following preprocessing steps were applied:

### 1. Handling Missing Values

* Missing values in numerical columns were handled using median values.
* Missing values in categorical columns were handled using the most frequent value.

### 2. Feature Selection

Unnecessary columns were removed:

* PassengerId
* Name
* Ticket
* Cabin

The final features used for training:

```
Pclass
Age
SibSp
Parch
Fare
Sex
Embarked
```

### 3. Encoding Categorical Features

Categorical variables were converted into numerical values using One-Hot Encoding.

Encoded features:

```
Sex_male
Embarked_Q
Embarked_S
```

Final model input features:

```
Pclass
Age
SibSp
Parch
Fare
Sex_male
Embarked_Q
Embarked_S
```

---

# Machine Learning Models

Several classification algorithms were tested:

* Decision Tree Classifier
* Random Forest Classifier
* Bagging Classifier
* Gradient Boosting Classifier
* XGBoost Classifier

The models were trained and evaluated using classification metrics.

---

# Model Used

The final selected model is:

## Bagging Classifier

Bagging improves model performance by training multiple Decision Trees and combining their predictions.

Parameters:

```python
BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=200,
    random_state=42
)
```

---

# Model Evaluation

The model performance was evaluated using:

### Accuracy

Measures the percentage of correct predictions.

### Classification Report

Includes:

* Precision
* Recall
* F1-score

Example:

```
Accuracy: 82%
```

---

# Model Saving

The trained model was saved using Joblib:

```
BaggingClassifier_model.pkl
```

Loading the model:

```python
import joblib

model = joblib.load("BaggingClassifier_model.pkl")
```

---

# Streamlit Application

A web application was developed using Streamlit to allow users to make predictions easily.

The user can enter:

* Passenger Class
* Age
* Number of siblings/spouses
* Number of parents/children
* Fare
* Sex
* Embarked

The application returns:

* Survival prediction
* Survival probability

Example:

```
Passenger Survived ✅
Survival Probability: 85.20%
```

---

# Project Structure

```
Titanic-Survival-Prediction/

│
├── app.py
├── BaggingClassifier_model.pkl
├── train.csv
├── test.csv
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

# Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Joblib
* Streamlit
* XGBoost

---

# Future Improvements

Possible improvements:

* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Feature engineering from passenger names and cabins
* Model deployment on cloud platforms

---

# Author

Machine Learning Project
Titanic Survival Prediction
