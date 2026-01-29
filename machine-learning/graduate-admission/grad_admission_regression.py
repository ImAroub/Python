import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load dataset
file_path = "Admission_Predict.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Step 2: Select columns
X = df[['GRE Score', 'TOEFL Score', 'CGPA', 'Research']]
y = df['Chance of Admit']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4 & 5: Pipeline (Standardization + Linear Regression)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
model = pipeline.named_steps['model']
print("\nModel Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")


# Step 6: Predict
y_pred = pipeline.predict(X_test)

# Step 7: Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"MAE : {mae:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R²  : {r2:.4f}")

# Step 8: Show comparison
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nActual vs Predicted (first 5 rows):")
print(comparison.head())

# Step 9: Plot relationship (CGPA vs Chance of Admit)
plt.scatter(df['CGPA'], df['Chance of Admit'], color='blue')
plt.xlabel('CGPA')
plt.ylabel('Chance of Admit')
plt.title('CGPA vs Chance of Admit')
plt.show()
