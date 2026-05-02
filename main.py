import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("train.csv")

print("Dataset Loaded ✅")

# -------------------------------
# 2. Show Columns (for debugging)
# -------------------------------
print("\nColumns in dataset:\n", df.columns)

# -------------------------------
# 3. Find Target Column
# -------------------------------
if "SalePrice" in df.columns:
    target = "SalePrice"
elif "price" in df.columns:
    target = "price"
else:
    raise Exception("❌ Target column not found (SalePrice/price)")

# -------------------------------
# 4. Select Numeric Data Only
# -------------------------------
df = df.select_dtypes(include=["number"])

# Drop missing values
df = df.dropna()

# -------------------------------
# 5. Split Features & Target
# -------------------------------
X = df.drop(target, axis=1)
y = df[target]

# -------------------------------
# 6. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 7. Linear Regression Model
# -------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# -------------------------------
# 8. Random Forest Model
# -------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# -------------------------------
# 9. Evaluation Function
# -------------------------------
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Results:")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

# Evaluate both models
evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# -------------------------------
# 10. Graph: Actual vs Predicted
# -------------------------------
plt.figure()
plt.scatter(y_test, y_pred_rf)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.show()

# -------------------------------
# 11. Feature Importance (RF)
# -------------------------------
importance = rf_model.feature_importances_
features = X.columns

feat_imp = pd.Series(importance, index=features).sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features")
plt.show()