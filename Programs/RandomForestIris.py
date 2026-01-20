from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["species"] = iris.target
data["species_name"] = data["species"].apply(lambda x: iris.target_names[x])

X = data[iris.feature_names]
y = data["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    oob_score=True,
    bootstrap=True
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (raw):\n")
print(cm)

print("\nOOB Score:", clf.oob_score_)
print("OOB Error:", 1 - clf.oob_score_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Iris Classification - Confusion Matrix")
plt.show()

example = clf.predict([[5.0, 3.2, 1.5, 0.2]])[0]
print("\nPredicted Species for Example:", iris.target_names[example])
