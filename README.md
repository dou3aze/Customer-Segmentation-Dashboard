# Customer Segmentation Analysis

A machine learning-based customer segmentation application built with Streamlit and scikit-learn for analyzing mall customer behavior patterns.

## 📚 Thesis Abstract

> **[Replace this with your thesis abstract]**
> 
> Include your thesis title, research objectives, methodology, key findings, and conclusions here. This should be 150-300 words summarizing your academic work.

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

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📊 Dataset

The project uses the Mall Customers dataset containing:
- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 🔍 Analysis Methodology

1. **Data Preprocessing**: Standardization of features
2. **Optimal Cluster Selection**: Elbow method and silhouette analysis
3. **Clustering**: K-Means and Hierarchical algorithms
4. **Evaluation**: Silhouette scores and visual inspection
5. **Interpretation**: Business insights from cluster characteristics

## 📈 Results

[Add your key findings and insights here]

## 👤 Author

**[Your Name]**
- Educational Institution: [Your University]
- Program: [Your Program/Degree]
- Year: 2026

## 📝 License

This project is part of academic work (PFE - Projet de Fin d'Études).

## 🙏 Acknowledgments

- [Your supervisor's name]
- [University/Department name]
- Dataset source: [If applicable]
