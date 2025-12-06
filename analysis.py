import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("train.csv")
print(df.head())
print(df.info())

# Drop irrelevant columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features & target
y = df["Survived"]
X = df.drop(columns=["Survived"])


# Train-test split - 20% for testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42
)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Train a logistic regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_scaled, y_train)

y_pred_logreg = log_reg.predict(X_test_scaled)
y_prob_logreg = log_reg.predict_proba(X_test_scaled)


# Train an SVM model
svm_model = SVC(probability=True) 
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
y_prob_svm = svm_model.predict_proba(X_test_scaled)

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)


# Evaluate models
def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Log Loss:", log_loss(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_logreg, y_prob_logreg)
evaluate_model("SVM", y_test, y_pred_svm, y_prob_svm)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)





# Plot confusion matrices
models = {
    "Logistic Regression": (y_test, y_pred_logreg),
    "SVM": (y_test, y_pred_svm),
    "Random Forest": (y_test, y_pred_rf),
}

plt.figure(figsize=(12, 4))

for i, (name, (y_true, y_pred)) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()