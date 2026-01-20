from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Train the Decision Tree using entropy (ID3-like)
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=12)
plt.title("Decision Tree on Iris Dataset (ID3 - Entropy)")
plt.show()
