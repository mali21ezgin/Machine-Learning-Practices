# 🏠 Housing Price Prediction Model

A machine learning project built with Python to predict housing prices using regression algorithms.

## 📊 Project Overview
This script performs data preprocessing, feature engineering, and rigorous hyperparameter optimization to compare the performance of two different machine learning models:
1. **Decision Tree Regressor** (Optimized for max depth, min samples split, and min samples leaf)
2. **K-Nearest Neighbors (KNN) Regressor** (Optimized for the best K value using scaled features)

## 🛠 Workflow & Pipeline
* **Data Loading & Inspection:** Reading the dataset using Pandas.
* **Preprocessing:** Handling categorical variables with One-Hot Encoding (`pd.get_dummies`) and splitting data.
* **Scaling:** Applying `StandardScaler` for distance-based algorithms like KNN.
* **Model Training & Tuning:** Automated loops to find the best hyperparameters based on Root Mean Squared Error (RMSE).
* **Evaluation:** Calculating RMSE and MSE metrics for both models.

## 🚀 How to Run
1. Clone the repository or download the files.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn
   ```
3. Run the script:
   ```bash
   python app.py
   ```