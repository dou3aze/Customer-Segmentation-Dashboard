# Customer Segmentation Analysis

A machine learning-based customer segmentation application built with Streamlit and scikit-learn for analyzing mall customer behavior patterns.

## 📚 Thesis Abstract

> This project addresses the challenge of understanding customer behavior in retail environments 
through data-driven analysis. Traditional approaches to customer classification rely on manual 
observation or intuition, which are inherently subjective and difficult to scale. In response to this 
limitation, this work proposes an automated segmentation system based on unsupervised machine 
learning applied to a publicly available retail dataset by applying K-Means clustering. The optimal 
number of clusters is determined using both the Elbow Method and the Silhouette Coefficient.
    A comparative analysis with Agglomerative Hierarchical Clustering is also conducted to validate
the robustness of the chosen approach. The resulting segments are presented through an 
interactive decision-support dashboard built with Streamlit, enabling non-technical users to explore 
customer profiles, filter data, and consult strategic recommendations. 

## 🎯 Project Overview

This project implements customer segmentation using K-Means and Hierarchical clustering algorithms to identify distinct customer groups based on their annual income and spending patterns. The interactive web application provides visualizations and insights for marketing strategy development.

## ✨ Features

- **Interactive Clustering**: K-Means and Agglomerative clustering algorithms
- **Multiple Visualizations**: 
  - Scatter plots with cluster assignments
  - Dendrograms for hierarchical clustering
  - Elbow method analysis
  - Silhouette analysis
- **Customer Segments Identified**:
  - High Income - High Spending (VIP)
  - High Income - Low Spending (At Risk)
  - Low Income - High Spending
  - Low Income - Low Spending
  - Average Customers

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations
- **Jupyter Notebook** - Exploratory analysis

## 📁 Project Structure

```
PFE/
├── app.py                 # Main Streamlit application
├── analysis.ipynb         # Exploratory data analysis
├── data/
│   └── Mall_Customers.csv # Customer dataset
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```
