import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset (change path to your file)
file_path = '/Users/ishan-college/Downloads/titanic-3f0b7484-dfd9-44be-a29e-278dd3165fc1 (1).csv'
df = pd.read_csv(file_path)

# Optional: print columns again for reference
print("Columns:", df.columns.tolist())

# Use lowercase feature names from your dataset
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
target = 'survived'

# Drop missing values in the selected columns + target
df = df[features + [target]].dropna()

# One-hot encode categorical features (sex, embarked)
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Features and labels
X = df.drop(columns=target)
y = df[target]

# Train Decision Tree with entropy (ID3)
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(model,
          feature_names=X.columns,
          class_names=["Not Survived", "Survived"],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Titanic Decision Tree (ID3 - Entropy)")
plt.tight_layout()
plt.show()
