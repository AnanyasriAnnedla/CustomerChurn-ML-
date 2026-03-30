import pandas as pd

# load dataset
data = pd.read_csv("churn.csv")

# drop unnecessary column
data = data.drop('customerID', axis=1)

# fix TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# convert target
data['Churn'] = data['Churn'].map({'Yes':1, 'No':0})

# encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# split data
X = data.drop('Churn', axis=1)
y = data['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

# evaluation
from sklearn.metrics import accuracy_score, confusion_matrix
print("Accuracy:", accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)