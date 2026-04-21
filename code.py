# ================================
# 1) Import Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ================================
# 2) Read Dataset
# ================================
df = pd.read_csv("Breast_cancer_data.csv")

# Remove index column (first column)
df = df.drop(df.columns[0], axis=1)

# Print basic info
print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# ================================
# 3) Handle Missing Values
# ================================
df = df.fillna(df.mean())

# ================================
# 4) EDA (Basic Visualizations)
# ================================
plt.figure(figsize=(6,4))
sns.countplot(x=df["diagnosis"])
plt.title("Diagnosis Count (0 = Benign, 1 = Malignant)")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 5) Split Features & Target
# ================================
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 6) Scaling
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 7) Train Models
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="rbf", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

    # ================================
    # (OPTIONAL 1) ROC Curve + AUC
    # ================================
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)

    print(f"AUC Score for {name}: {auc_score:.4f}")

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.show()


# ================================
# 8) Compare Model Accuracies
# ================================
print("\n\n=== Model Comparison ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print("\nBest Model is:", best_model)

# ================================
# 9) (OPTIONAL 2) GridSearchCV for Random Forest
# ================================
print("\n===== Running Grid Search for Random Forest =====")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)
# Evaluate tuned model
best_rf = grid.best_estimator_
y_pred_grid = best_rf.predict(X_test)

print("\n===== Tuned Random Forest Evaluation =====")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print(classification_report(y_test, y_pred_grid))

cm = confusion_matrix(y_test, y_pred_grid)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("Tuned Random Forest - Confusion Matrix")
plt.show()