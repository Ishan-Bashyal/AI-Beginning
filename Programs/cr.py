import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('/Users/ishan-college/Downloads/clash_royale_cards.csv')

# ---- Select MORE THAN 2 features ----
X = df[["Win Rate", "Usage", "Usage Change"]]

# Select target variable
y = df["Win Rate Change"]
# -------------------------------------

# Train model
model = LinearRegression()
model.fit(X, y)

# Print intercept and coefficients
print("Intercept (β₀):", model.intercept_)
print("\nCoefficients (β values):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# ---- Make Predictions ----
print("\n--- Prediction Example ---")

try:
    win_rate = float(input("Enter Win Rate: "))
    usage = float(input("Enter Usage: "))
    usage_change = float(input("Enter Usage Change: "))

    input_df = pd.DataFrame([[win_rate, usage, usage_change]],
                            columns=["Win Rate", "Usage", "Usage Change"])

    prediction = model.predict(input_df)

    print(f"\nPredicted Win Rate Change: {prediction[0]:.4f}")

except ValueError:
    print("Invalid input. Please enter numeric values.")
