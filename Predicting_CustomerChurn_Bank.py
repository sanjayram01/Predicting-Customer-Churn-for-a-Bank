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
df = pd.read_csv("Final_BankChurners.csv")
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


#%%[markdown]
# # EDA(Exploratory Data Analysis)
# EDA 1: 






#%%
# EDA 5: Analysis of higher probability of low-income level customer churning by high credit limit
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
