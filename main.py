import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. INCARCARE SI CURATARE DATE
df = pd.read_csv('cardekho.csv')

# Eliminam coloana seats
if 'seats' in df.columns:
    df = df.drop(columns=['seats'])
df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Completare valori lipsa cu media
cols_to_fix = ['mileage(km/ltr/kg)', 'engine', 'max_power']
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].mean())

# 2. ELIMINARE ANOMALII (IQR)
Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['selling_price'] >= (Q1 - 1.5 * IQR)) & (df['selling_price'] <= (Q3 + 1.5 * IQR))]

# 3. ANALIZA INDICATORILOR (CORELAȚIE & INFO)
# Encoding pentru coloanele text
df_encoded = df.copy()
le = LabelEncoder()
categorice = df_encoded.select_dtypes(include=['object', 'string']).columns
for col in categorice:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

print("Corelatie cu Pretul")
print(df_encoded.corr()['selling_price'].sort_values(ascending=False).head(5))

# 4. ANTRENRE SI OPTIMIZARE MODELE
X = df_encoded.drop('selling_price', axis=1)
y = df_encoded['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evalueaza(nume, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{nume}] MAE: {mae:.2f}, R2: {r2:.4f}")
    return mae, r2

# Modelele
models_data = {}

# A. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
models_data['Linear Regression'] = evalueaza("Linear Regression", y_test, lr.predict(X_test))

# B. Random Forest
print("\nOptimizare Random Forest")
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                       {'n_estimators': [100, 150], 'max_depth': [10, 20]},
                       cv=3).fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
models_data['Random Forest'] = evalueaza("Random Forest", y_test, best_rf.predict(X_test))

# C. Gradient Boosting
print("Optimizare Gradient Boosting")
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42),
                       {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05]},
                       cv=3).fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
models_data['Gradient Boosting'] = evalueaza("Gradient Boosting", y_test, best_gb.predict(X_test))

# 5. ALEGERE MODEL FINAL
print("\nREZULTAT FINAL")
for m, res in models_data.items():
    print(f"{m}: R2 Score = {res[1]:.4f}")

print("\nModelul ales: Random Forest (datorita stabilitatii si scorului R2 ridicat)")