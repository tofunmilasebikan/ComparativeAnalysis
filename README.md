# Classification Model Comparison (Titanic Dataset)

This project compares three machine learning models—**Logistic Regression**, **Support Vector Machine (SVM)**, and **Random Forest**—using the kaggle dataset (`train.csv`).

The goal is to evaluate how different algorithms perform on the same classification task using consistent preprocessing and evaluation metrics.

---

## 📊 Models & Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~0.81 |
| SVM | ~0.82 |
| Random Forest | ~0.81 |

SVM achieved the highest accuracy, while Logistic Regression produced well-calibrated probability outputs (low log loss).

---

## 🧹 Preprocessing Summary
- Removed irrelevant columns  
- Filled missing values  
- One-hot encoded categorical data  
- Scaled features for LR and SVM  

---

## 📈 Visuals
The project includes confusion matrix heatmaps for each model.

---

## 🔧 How to Run
