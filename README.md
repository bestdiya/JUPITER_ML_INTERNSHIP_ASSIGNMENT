# JUPITER_ML_INTERNSHIP_ASSIGNMENT
# Credit Score Movement Prediction

## Project Overview

This project develops a machine learning model to predict the movement of a customer's credit score (increase, stable, or decrease). Understanding credit score dynamics is critical for financial institutions for accurate risk assessment, targeted product offerings, and proactive customer engagement.

This repository contains the code and analysis for an end-to-end machine learning pipeline, including synthetic data generation, exploratory data analysis (EDA), data preprocessing, model development, evaluation, explainability, and translation of findings into actionable business insights.

## Table of Contents

1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4.  [Methodology](#methodology)
    *   [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
    *   [Handling Class Imbalance](#handling-class-imbalance)
    *   [Model Development & Evaluation](#model-development--evaluation)
    *   [Model Explainability](#model-explainability)
5.  [Key Findings & Insights](#key-findings--insights)
6.  [Business Recommendations](#business-recommendations)
7.  [Repository Structure](#repository-structure)
8.  [How to Run the Code](#how-to-run-the-code)
9.  [Dependencies](#dependencies)
10. [License](#license)

## Project Goal

The primary objective of this project is to build and evaluate machine learning models that can accurately predict whether a customer's credit score is likely to increase, remain stable, or decrease based on their demographic, financial, and credit bureau attributes.

## Dataset

Due to the sensitive nature of real credit data, this project utilizes a **synthetically generated dataset**.

*   **Size:** 25,000 customer records.
*   **Features:** The dataset includes a comprehensive set of features simulating real-world data points commonly found in credit profiles, categorized as:
    *   Basic Demographics (e.g., Age, Gender, Location)
    *   Financial Behavior (e.g., Monthly Income, EMI Outflow, Credit Utilization, Repayment History)
    *   Credit Bureau Features (e.g., Total Trade Lines, Days Past Due, Hard Inquiries, Age of Accounts)
*   **Target Variable:** `target_credit_score_movement` (Categorical: 'increase', 'stable', 'decrease'). The target was generated based on a simulated business logic incorporating the influence of various features on credit score movement.

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the dataset's structure, feature distributions, and relationships with the target variable. Key insights from the EDA informed the subsequent feature engineering and modeling steps.

*   Identified the distribution of credit score movement categories.
*   Analyzed the relationship between critical features (e.g., Credit Utilization, Repayment History, DPD) and the target.
*   Revealed the class imbalance issue in the target variable.

## Methodology

The project follows a standard machine learning workflow:

### Feature Engineering & Preprocessing

*   Created derived features such as `debt_to_income_ratio` and categorized continuous variables into 'buckets' (e.g., `utilization_bucket`, `income_bucket`).
*   Encoded categorical features using `LabelEncoder`.
*   Scaled numerical features using `StandardScaler` to normalize their range.

### Handling Class Imbalance

*   Addressed the imbalance in the target variable using **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the class distribution in the training data.

### Model Development & Evaluation

*   The balanced dataset was split into training (80%) and testing (20%) sets using stratification.
*   A range of classification models were trained and evaluated:
    *   Logistic Regression
    *   Random Forest Classifier
    *   XGBoost Classifier
    *   Decision Tree
    *   K-Nearest Neighbors
    *   Naive Bayes
    *   Gradient Boosting Classifier
    *   LightGBM Classifier
*   Model performance was assessed using:
    *   Accuracy
    *   Weighted F1-Score (primary metric due to classification problem)
    *   Confusion Matrices
    *   AUC-ROC curves and scores (using the One-vs-Rest approach for multi-class evaluation)


## Key Findings & Insights

*   Features strongly correlated with credit score movement were identified, reinforcing known credit risk factors.
*   Key drivers for credit score increase were found to be excellent repayment history, low credit utilization, and absence of DPD.
*   Significant factors contributing to score decrease included high DPD, high credit utilization, and high total missed payments.
*   Distinct profiles for high-risk and high-opportunity customer segments were identified based on these key features.

## Business Recommendations

Based on the analysis and model insights, actionable recommendations were developed for financial institutions:

*   **High-Risk Interventions:** Targeted programs like credit counseling, utilization alerts, and flexible EMI restructuring.
*   **High-Opportunity Growth:** Offers like credit limit increases, premium product cross-selling, and reward programs for good behavior.
*   **Monitoring & Retention:** Implementing dashboards, personalized financial tips, and proactive communication.
*   **Product Features:** Developing tools like credit score simulators and automated savings features.

