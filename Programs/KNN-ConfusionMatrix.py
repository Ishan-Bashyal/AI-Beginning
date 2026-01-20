import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

TP = 30
FN = 20
FP = 40
TN = 910

cm = [[TN, FP],
      [FN, TP]]

cm_df = pd.DataFrame(cm, 
                     index=["Actual Legitimate", "Actual Fraudulent"],
                     columns=["Predicted Legitimate", "Predicted Fraudulent"])

print("Confusion Matrix:")
print(cm_df)

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Fraud Detection Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

