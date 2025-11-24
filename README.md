# 🎯 TikTok Claim vs. Opinion Classification Pipeline

## 📝 Executive Summary

This repository documents the comprehensive data analytics and machine learning pipeline developed over six stages of the Google Advanced Data Analytics Professional Certificate. The goal is to address the critical business problem of classifying content on the TikTok platform into **"Claim"** or **"Opinion"** categories based on video metadata and engagement metrics.

The entire project demonstrates end-to-end proficiency across the data science lifecycle, from initial planning and exploratory data analysis (EDA) to statistical inference, regression modeling, and pipeline development.

---

## 🛣️ Project Architecture and Progression

This project is structured chronologically, following the specialization's curriculum and the **PACE (Plan, Analyze, Construct, Execute) Workflow**. Each folder represents a distinct phase of the pipeline development, collectively addressing the core classification challenge.

| Folder | Course Focus | Key Technical Deliverables |
| :--- | :--- | :--- |
| **`01_Foundations_PACE_Proposal`** | **Project Planning & Strategy** | Initial **Project Proposal** defining scope, metrics, and business rationale. Established the analytical direction for claim detection. |
| **`02_Python_for_DA`** | **Data Ingestion & Initial Analysis** | Python (Pandas) scripts for data loading, quality assessment, and identifying significant differences in **engagement trends** (e.g., Claim views vs. Opinion views). |
| **`03_EDA_and_Visualization`** | **Exploratory Data Analysis (EDA)** | Deep dive into data distributions. Creation of professional visualizations using **Matplotlib/Seaborn** and **Tableau** to uncover factors driving virality (e.g., Author Ban Status). |
| **`04_Statistical_Inference`** | **Hypothesis Testing & Validation** | Application of inferential statistics to formally validate observed correlations and differences in group means, providing a statistical basis for feature importance. |
| **`05_Regression_Modeling`** | **Predictive Model Prototyping** | Implementation of **Logistic Regression** and other models to generate initial classification predictions. Focused on model training, selection, and performance evaluation. |
| **`06_ML_Pipelines_Feature_Eng`** | **Workflow Automation & Optimization** | Development of a scalable, reproducible workflow covering data preprocessing, advanced **Feature Engineering**, and integration into an automated ML pipeline structure. |

---

## 📈 Key Discovery Highlight (Course 2 Insight)

A major finding established early in the project (Course 2) revealed a significant correlation between `claim_status` and content visibility:

* **Claim Videos** recorded $\approx 501K$ average views.
* **Opinion Videos** recorded $\approx 4.9K$ average views.

> **Result:** Claim videos exhibit a viewership **magnitude two orders higher** than opinion videos, confirming that simple view count is a powerful predictive feature that must be incorporated into the classification model.

---

## 🛠️ Technology Stack

| Category | Tools & Libraries Used |
| :--- | :--- |
| **Programming** | Python |
| **Core Libraries** | Pandas, NumPy (Data Manipulation) |
| **Modeling** | Scikit-learn, Statsmodels (Machine Learning, Statistics) |
| **Visualization** | Matplotlib, Seaborn, Tableau (Data Storytelling) |
| **Methodology** | PACE Framework (Plan, Analyze, Construct, Execute) |

---

## 🔗 Related Project

The final **Capstone Project** that synthesized all these phases into a complete, deployable solution is available separately:

> [**https://github.com/Shanekhan/Salifort-Motors-Employee-Retention**]

---

