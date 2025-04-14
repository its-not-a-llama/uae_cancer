"""
Abstract:
This project explores the application of multinomial logistic regression to predict cancer outcomes using a dataset of cancer patients. The dataset includes various demographic and clinical features. Despite the synthetic nature of the data, the project demonstrates key data science techniques, including data cleaning, feature engineering, and model evaluation.

Introduction:
The goal of this project is to predict the outcome of cancer patients based on demographic and clinical features. The dataset includes information such as age, gender, cancer type, and treatment details. By applying multinomial logistic regression, we aim to identify significant predictors of patient outcomes.

Data Cleaning and Preprocessing:
"""
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import janitor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load and clean data
data = pd.read_csv("uae_cancer.csv")
data = data.clean_names(case_type="snake")

# Drop unnecessary columns
data.drop(columns=["patient_id", "primary_physician"], inplace=True)

# Convert relevant columns to categorical
categorical_cols = ["gender", "nationality", "emirate", "cancer_type", "cancer_stage", 
                    "treatment_type", "cause_of_death", "smoking_status", "comorbidities", "ethnicity"]
data[categorical_cols] = data[categorical_cols].astype("category")

# Convert date columns to datetime
date_cols = ["diagnosis_date", "treatment_start_date", "death_date"]
data[date_cols] = data[date_cols].apply(pd.to_datetime)

# Filter data based on cutoff date
cutoff_date = pd.to_datetime("2025-02-28")
data = data[(data['death_date'].isna()) | (data['death_date'] <= cutoff_date)]

# Calculate age of diagnosis
birth_year = cutoff_date.year - data["age"]
data["age_of_diagnosis"] = data["diagnosis_date"].dt.year - birth_year

# Calculate BMI and drop weight and height
data["bmi"] = data["weight"] / ((data["height"] * 0.01) ** 2)
data.drop(columns=["weight", "height"], inplace=True)

"""
Exploratory Data Analysis:
"""

# Visualize outcome distribution
sns.histplot(data=data, x="outcome")
plt.title('Outcome Distribution')
plt.show()

# Insight: The distribution of outcomes shows a higher frequency of certain outcomes, which may affect model performance.

# Visualize cancer types by stage using a heatmap
crosstab = pd.crosstab(data['cancer_stage'], data['cancer_type'])
plt.figure(figsize=(14, 8))
sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Cancer Types by Stage')
plt.xlabel('Cancer Type')
plt.ylabel('Cancer Stage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Insight: The heatmap reveals patterns in cancer type distribution across stages, which could inform feature selection.

"""
Modeling:
"""

# Prepare data for modeling
X = data[['age', 'gender', 'nationality', 'emirate', 'cancer_type', 
          'cancer_stage', 'treatment_type', 'hospital', 'smoking_status', 
          'comorbidities', 'ethnicity', 'age_of_diagnosis', 'bmi']]
y = data['outcome']

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Normalize the features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

"""
Results and Discussion:
"""

# Predict and evaluate the model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Insight: The model shows varying performance across different outcomes, with better accuracy for certain classes.

# Analyze feature importance
coefficients = pd.DataFrame(model.coef_, columns=X.columns, index=model.classes_)
mean_abs_coefficients = coefficients.abs().mean(axis=0)
sorted_coefficients = mean_abs_coefficients.sort_values(ascending=False)
print("Feature Importance:")
print(sorted_coefficients)

# Insight: Feature importance analysis highlights which predictors have the most influence on the model's decisions.

"""
Limitations:
- The dataset is synthetic, which may not accurately reflect real-world patterns and relationships.
- The model's performance is limited by the quality and authenticity of the data.
- Further validation with authentic data is necessary to confirm the findings.

Conclusion:
This project demonstrates the application of multinomial logistic regression to a synthetic cancer dataset. While the model provides insights into potential predictors of cancer outcomes, the synthetic nature of the data limits the generalizability of the results. Future work should focus on validating these findings with real-world data.
"""