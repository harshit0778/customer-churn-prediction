# Customer Churn Prediction Project

## Overview
This project predicts whether a telecom customer is likely to churn using machine learning.  
It uses a complete preprocessing and modeling pipeline so that raw customer data can be directly used for predictions.

The goal is to identify churn customers early so that retention actions can be taken.
![alt text](image-1.png)

---

## Dataset
**Source:** Telco Customer Churn Dataset (Kaggle)

The dataset contains customer demographics, service usage, billing information, and a churn label.

---

## Project Structure
```
customer_churn/
│
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebook/
│ └── churn_eda_modeling.ipynb
|
├── plots/
│ └── *seaborn plots*
│
├── model/
│ ├── churn_model.pkl
│ ├── new_pred.ipynb
│ └── new_customer.txt
│
├── src/
│ └── churn_model.py
│
├── README.md
└── requirements.txt
```
---

## Plots
corr matrix plot:
![alt text](plots/Corr_matrix.png)
most important features RFC:
![alt text](plots/Most_imp_features.png)

---

## Approach
- Cleaned and prepared the dataset
- Converted `TotalCharges` to numeric and handled missing values
- Standardized service-related categories for consistency
- Built a preprocessing pipeline using:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- Trained a Logistic Regression model with class balancing
- Used probability-based predictions and threshold tuning
- Evaluated the model using precision, recall, F1-score, confusion matrix, and ROC-AUC
- Saved the complete pipeline for reuse on new data

---

## Model Pipeline
The model is built using a single pipeline that includes:
- Data preprocessing (scaling and encoding)
- Logistic Regression classifier

Because preprocessing is inside the pipeline, **raw customer data can be passed directly to the model for prediction**.

---

## Results
- Logistic Regression achieved a ROC-AUC of approximately 0.85
- High recall for churn customers (approximately 0.86)
- Model prioritizes reducing missed churn customers

---

## Making Predictions on New Data
A saved pipeline model is used to predict churn for new customers.

Steps:
Explain:
1. Load the saved model
2. Provide new customer data in the same raw format as training
3. Apply the same basic data cleaning (service value replacements)
4. Predict churn probability and class

Prediction examples are included in:
- `model/new_pred.ipynb`
- `model/new_customer.txt`

---

## Technologies Used
Python, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## How to Run
1. Install dependencies:

2. Run the notebook for EDA and training:

3. Use the saved model for new predictions:

---

## Conclusion
This project demonstrates an end-to-end customer churn prediction system using a production-style machine learning pipeline.  
The focus is on business impact by minimizing missed churn customers and enabling direct predictions on new data.

---

## 👤 Author

Harshit Rathore
Aspiring Data Analyst 
