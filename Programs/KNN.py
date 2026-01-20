import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('/Users/ishan-college/Downloads/titanic-3f0b7484-dfd9-44be-a29e-278dd3165fc1.csv')

print(df.head())

features = ['pclass', 'sex', 'age', 'fare', 'embarked']
df = df[features + ['survived']]

df = df.dropna()

df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[features]
y = df['survived']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

accuracy_list = []
for k in range(1, 12):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_val_pred = knn.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    accuracy_list.append(acc)
    print(f"K={k}, Validation Accuracy={acc:.4f}")

best_k = accuracy_list.index(max(accuracy_list)) + 1
print("\nBest K value based on validation set:", best_k)

X_final_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_final_train_scaled, y_train)

y_test_pred = final_knn.predict(X_test_scaled)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nFinal Classification Report:\n", classification_report(y_test, y_test_pred))
