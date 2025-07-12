# 🫀 Heart Disease Prediction with Machine Learning

This project applies machine learning to predict the presence of heart disease using clinical patient data. The goal is to assist early detection by comparing multiple models, evaluating performance with cross-validation, and interpreting model behavior with feature importance and confusion matrices.

---

## 📌 Problem Statement

Heart disease remains a leading cause of death globally. Early and accurate detection can save lives. This project explores how ML models can classify patients at risk using available health data.

---

## 🚀 What I Did

- ✅ Cleaned and prepared clinical data for modeling.
- ✅ Trained and tuned four ML models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - CatBoost Classifier
- ✅ Evaluated models with:
  - Test metrics (Accuracy, F1, Precision, Recall, AUC)
  - Cross-validation scores (5-fold)
  - Confusion Matrices
  - ROC Curves
  - Sensitivity and Specificity
- ✅ Analyzed overfitting using train vs test performance.
- ✅ Interpreted model behavior via feature importance visualizations.

---

## 📊 Model Performance Snapshot

| Model               | Accuracy | F1 Score | AUC   | CV F1 | CV AUC |
|--------------------|----------|----------|-------|-------|--------|
| Logistic Regression| 0.8852   | 0.8923   | 0.9246| 0.8559| 0.8913 |
| KNN                | 0.7049   | 0.7097   | 0.8147| 0.7466| 0.7226 |
| Random Forest      | 0.8689   | 0.8788   | 0.9353| 0.8418| 0.9079 |
| CatBoost           | 0.8689   | 0.8788   | 0.9127| 0.8594| 0.9108 |

> ✅ **Best Model**: Logistic Regression provided the best balance between test and CV metrics with strong recall and AUC — ideal for clinical deployment.

---

## 📈 Key Visuals

- ROC Curves for model comparison  
- Confusion Matrices with class labels  
- Bar plots of test vs CV performance  
- Feature importances (CatBoost & Random Forest)  
- Sensitivity vs Specificity breakdown  

---

## 🛠 Tech Stack

- Python: `pandas`, `numpy`, `scikit-learn`, `catboost`, `matplotlib`, `seaborn`
- Jupyter Notebook (Exported to HTML for sharing)

---

## 💡 Key Takeaways

- Combining CV and test metrics improves model selection confidence  
- Analyzing overfitting with test vs CV performance helped identify the most generalizable model  
- Logistic Regression and CatBoost both performed well  
- Feature importance increases transparency for clinical ML use  

---

## 📂 How to Run

```bash
git clone https://github.com/albe290/heart-disease-prediction.git
cd heart-disease-prediction
jupyter notebook Heart-Disease-Prediction.ipynb
