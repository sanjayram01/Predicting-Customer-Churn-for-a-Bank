#%%[markdown]
# # Project : Predicting Customer Churn for a Bank
#
# Team Members: <br>
# Brian Kim <br>
# Kusum Sai Chowdary Sannapaneni <br>
# Sanjayram Raja Srinivasan <br>
# Shashank Shivakumar
#
# Research Topic: <br>
# Customer retention is vital for any business, and the banking sector is no exception. With the rise of digital banking and increased competition, understanding why customers leave and predicting potential churn is crucial. Our team aims to delve deep into the factors influencing customer churn for a bank. By using machine learning and data analytics techniques, we want to predict the likelihood of a customer leaving the bank in the upcoming months. This prediction, backed by insights from the data, can help the bank in formulating effective retention strategies.
#
# Objective: <br>
# Our primary objective is to build a predictive model that can accurately identify customers at the highest risk of churning in the next three months. Additionally, we aim to analyze and interpret the data to discover the main drivers of customer churn. These insights will provide the bank with actionable recommendations to enhance customer satisfaction and loyalty.
#
# SMART Question:
# 1. Is the interaction between Months_on_book and Total_Relationship_Count relevant to the bank's strategic objectives, particularly in the context of diminishing customer attrition? <br>
# 2. Is there a measurable trend that shows how Months_on_book affects customer attrition over time, perhaps by analyzing attrition rates in different periods? <br>
# 3. Is there a discernible pattern where clients in higher income categories exhibit a lower attrition rate, and if so, what strategies can be formulated to enhance client retention among those in lower income categories? <br>
# 4. Can we predict the likelihood of a customer leaving the bank in the next 3 months based on their past transaction behavior and demographic information? <br>
# 5. If a customer has a low-income level but a high credit limit, is there a higher probability of them churning? <br>
#
# Modeling Methods: <br>
# We propose to start with a baseline model using Logistic Regression to establish initial performance metrics. Subsequent models will include more complex algorithms like Random Forest and Gradient Boosted Trees. We will also consider neural networks if the dataset size and feature engineering justify its use. Model evaluation will be done using metrics like accuracy, precision, recall, F1-score, and the AUC-ROC curve. Feature importance and model interpretability will also be integral parts of our analysis.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# Data info
df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")
df.head(5)

#%%[markdown]
# # Data Preprocessing

# Checking number of missing values in each column
missing_df = df.isnull().sum()
print(missing_df)


#%%
# Summary statistics
df.describe()

#%%
# checking datatypes and null values
dataset.info()

#%%[markdown]
# # EDA(Exploratory Data Analysis)
# Histogram of overall distro of age

# Create a histogram
plt.hist(df['Customer_Age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.grid(True)
plt.show()

#%%
# Plot stacked histogram on age for male and female stacked together
# Separate "age" values for males and females
male_age = df[df['Gender'] == 'M']['Customer_Age']
female_age = df[df['Gender'] == 'F']['Customer_Age']

# Create a stacked histogram
plt.hist([male_age, female_age], bins=20, stacked=True, label=['Male', 'Female'], edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Stacked Histogram of Age for Males and Females')
plt.legend()
plt.show()

#%%
# Month on book Box plot
import seaborn as sns
import matplotlib.pyplot as plt

# Create a box plot for Months_on_book
plt.figure(figsize=(12, 6))
sns.boxplot(x='Months_on_book', data=df)

# Set plot labels and title
plt.xlabel('Months on Book')
plt.title('Box Plot of Months on Book')

# Show the plot
plt.show()



#%%
# EDA: correlation analysis
import pandas as pd

# Select specific columns
selected_columns = ['Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Credit_Limit', 'Total_Trans_Amt', 'Avg_Utilization_Ratio']

# Create a new DataFrame with only the selected columns
selected_df = df[selected_columns]

# One-hot encode categorical variables
selected_df_encoded = pd.get_dummies(selected_df)

# Calculate correlation for the selected columns
correlation_matrix = selected_df_encoded.corr()

# Display the correlation matrix
print(correlation_matrix)

# visualization heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data=correlation_matrix, annot=False, fmt='.2f')
plt.title("Heatmap of data in Customer_Churn")
plt.show()


#%%
# EDA - Analysis of higher probability of low-income level customer churning by high credit limit
# Analysis of correlation

# calculate the correlation coefficient
df_numeric = df.select_dtypes(include=[np.number])
df_cor = df_numeric.corr()

#%%
# visualization heatmap
plt.figure(figsize=(7,6))
sns.heatmap(data=df_cor, annot=False, fmt='.2f')
plt.title("Heatmap of data in Customer_Churn")
plt.show()


# %%
df = pd.read_csv("Final_BankChurners.csv")

# Encode categorical variables
df = pd.get_dummies(df, columns=["Attrition_Flag", "Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"])

# Remove irrelevant columns
df = df.drop(["CLIENTNUM"], axis=1)  # You may need to adjust this based on your specific data

# Calculate correlation
correlation_matrix = df.corr()

# Display the correlation matrix
print(correlation_matrix)

df_cor = df_numeric.corr()

#%%
# visualization heatmap
plt.figure(figsize=(7,6))
sns.heatmap(data=correlation_matrix, annot=False, fmt='.2f')
plt.title("Heatmap of data in Customer_Churn")
plt.show()

# %%[markdown]
# Q5 If a customer has a low-income level but a high credit limit, is there a higher probability of them churning?
# Logistic regression
# Data info
df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Explore the data
print(df.info())
print(df.describe())

# Visualize the relationship between income level, credit limit, and churn
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# You can use different visualization techniques, such as a scatter plot or box plot
sns.scatterplot(x='Income_Category', y='Credit_Limit', hue='Attrition_Flag', data=df)
plt.title('Income vs. Credit Limit by Churn Status')
plt.xlabel('Income Category')
plt.ylabel('Credit Limit')
plt.show()

# Convert categorical variables to numerical for modeling
df['Income_Category'] = df['Income_Category'].astype('category').cat.codes

# Build a simple logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%[markdown]
# Decision Tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")


# Convert categorical variables to numerical for modeling
le = LabelEncoder()
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit the decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%[markdown]
# Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Change the import statement
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")

# Convert categorical variables to numerical for modeling
le = LabelEncoder()
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit the random forest model
# Change DecisionTreeClassifier to RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %%[markdown]
# Comparison between Decision Tree and Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")

# Convert categorical variables to numerical for modeling
le = LabelEncoder()
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate and compare the models
print("Decision Tree Performance:")
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("Accuracy:", accuracy_score(y_test, dt_pred))

print("\nRandom Forest Performance:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize the labels for ROC curve calculation
y_test_bin = label_binarize(y_test, classes=[0, 1])

# Use OneVsRestClassifier for multi-class ROC curve
classifier = OneVsRestClassifier(dt_model)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]  # Probability of the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree - Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# %%[markdown]
# Parameter Tuning for Random Forest 1
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")


# Convert categorical variables to numerical for modeling
le = LabelEncoder()
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
print("Best Parameters:", best_params)
print("\nBest Model Performance:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# %%[markdown]
# Parameter Tuning for Random Forest 2
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("/Users/brian/Documents/GitHub/BankChurners/DATS6103_10_Group_5/Final_BankChurners.csv")


# Convert categorical variables to numerical for modeling
le = LabelEncoder()
df['Income_Category'] = le.fit_transform(df['Income_Category'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
df['Card_Category'] = le.fit_transform(df['Card_Category'])
df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

# Select relevant features and target variable
features = ['Income_Category', 'Credit_Limit', 'Total_Trans_Amt', 'Avg_Utilization_Ratio']
target = 'Attrition_Flag'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predictions with the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
print("Best Parameters:", best_params)
print("\nBest Model Performance:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# %%
# AUC-ROC for Random Forest
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Binarize the labels for ROC curve calculation
y_test_bin = label_binarize(y_test, classes=[0, 1])

# Use OneVsRestClassifier for multi-class ROC curve
classifier = OneVsRestClassifier(best_model)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)[:, 1]  # Probability of the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test_bin, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# %%
