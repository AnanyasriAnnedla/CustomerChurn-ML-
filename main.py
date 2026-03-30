import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("churn.csv")

data = data.drop('customerID', axis=1)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

data = pd.get_dummies(data, drop_first=True)

X = data.drop('Churn', axis=1)
y = data['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

plt.figure()
sns.countplot(x=y)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

plt.figure()
sns.boxplot(x=y, y=data['tenure'])
plt.title("Tenure vs Churn")
plt.savefig("tenure_vs_churn.png")
plt.show()