# Breast Cancer Prediction using Machine Learning 

##  Project Overview
This project focuses on building a diagnostic tool to classify breast tumors as **Malignant** or **Benign**. Using the Wisconsin Breast Cancer Dataset, I implemented a full Machine Learning pipeline—from exploratory data analysis to model optimization—achieving high accuracy in medical classification.

## Methodology & Workflow

### 1. Exploratory Data Analysis (EDA)
Understanding the data was the first step. I analyzed the distribution of classes and the correlation between different clinical features.

* **Class Distribution:** Used a Count Plot to check for data balance.
* **Correlation Analysis:** Identified key features using a Heatmap.

<p align="center">
  <img src="/images/target_distribution.png" width="45%" title="Target Distribution" />
  <img src="/images/Correlation_Heatmap.png" width="45%" title="Correlation Heatmap" />
</p>

### 2. Model Implementation & Tuning
I compared two powerful classification algorithms:
- **Logistic Regression:** A robust baseline for binary classification.
- **Random Forest Classifier:** An ensemble method optimized using **GridSearchCV** for the best hyperparameters.

## Performance Evaluation
The models were evaluated using clinical-standard metrics to ensure reliability.

### Confusion Matrices
These matrices show the precision of our models in identifying true positives and minimizing false negatives.
<p align="center">
  <img src="/images/confusion_matrix_RandomForest.png" width="45%" />
  <img src="/images/confusion_matrix_Tuned.png" width="45%" />
</p>

### ROC Curves 
The ROC curves illustrate the trade-off between sensitivity and specificity for model.
<p align="center">
  <img src="/images/ROC_Curve.png" width="45%" />
</p>

## Technologies Used
- **Language:** Python
- **Libraries:** Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib.
- **Tools:** Jupyter Notebook / Google Colab.

##  Repository Structure
- `code.py`: Full source code.
- `Breast_cancer_data.csv`: Dataset in CSV format.
- `images/`: Visualization plots and performance charts.
- `Report ML Predict Breast Cancer.pdf`: Comprehensive project report.

-
