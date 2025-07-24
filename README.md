#  Product Return Pattern Analysis for E-Commerce

A complete end-to-end Data Analyst + Machine Learning project to analyze and predict product return behavior in an e-commerce setting using Python, Streamlit, and scikit-learn.


[![Streamlit App](https://img.shields.io/badge/Live-App-green?logo=streamlit)](https://appuctreturnanalysis-dkvbwjlixmomazkgfbaivg.streamlit.app/)

---

##  Project Overview

This project focuses on analyzing historical product return data to understand key return patterns and predict the probability of a product being returned using machine learning.

### 🔍 Problem Statement
In the e-commerce industry, product returns directly impact customer satisfaction, logistics costs, and seller reputation. By understanding return patterns, we can:
- Reduce return rates
- Identify high-risk categories or users
- Take proactive measures in UI, packaging, or logistics

---

##  Dashboard Preview

An interactive web dashboard built using **Streamlit + Plotly** that allows:
- Filtering return trends by **category & date**
- Viewing **monthly return rates**
- Predicting return **likelihood** based on product/user inputs
- Downloading reports and prediction history

---

##  Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python |
| EDA & Cleaning | Pandas, NumPy |
| Visualizations | Plotly, Streamlit |
| Machine Learning | scikit-learn (Logistic Regression) |
| Model Persistence | Joblib |
| App Framework | Streamlit |
| Deployment | (Streamlit Cloud or Localhost) |

---

##  Folder Structure

ProductReturnAnalysis/
│
├── data/ # Raw & cleaned datasets
├── models/ # Trained model & encoders
├── dashboard/ # Streamlit app
├── scripts/ # ML training script
├── reports/ # (optional) PDF or CSV summaries
└── README.md # Project documentation

---

##  Features Implemented

- 📅 **Date & category filtering**
- 📈 **Monthly return trend charts**
- 📊 **Return rate table by category**
- 🔮 **ML-based return prediction**
- 💾 **Prediction history tracking**
- 📥 **Downloadable CSV reports**

---

##  Machine Learning Details

- Model: `Logistic Regression`
- Input features:
  - Product: category, price, discount, quantity
  - User: age, gender
  - Order: shipping method, payment method, return reason
- Encoded using: `LabelEncoder` + `StandardScaler`
- Evaluation: `Train/Test Split` + `Classification Report`

---

## 📦 Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/ProductReturnAnalysis.git

Sample Input for Prediction
Field	              Example
Product Category	  Electronics
Product Price	      899.0
Order Quantity     	   1
User Age	          28
Gender	              Female
Return Reason	      Damaged
Payment Method	      Credit Card
Shipping Method	      Express
Discount Applied	  10%