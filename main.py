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
# Incarcam setul de date brut intr-un DataFrame Pandas
df = pd.read_csv('cardekho.csv')

# Eliminam coloana 'seats' fiind considerata irelevanta pentru predictie
if 'seats' in df.columns:
    df = df.drop(columns=['seats'])

# Extragem doar valorile numerice din coloanele de tip text
# Folosim expresii regulate si prefixul r'' pentru a evita erorile de tip escape sequence
df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Gestionarea valorilor lipsa: inlocuim celulele goale cu media coloanei, pentru ca majoritatea algoritmilor nu pot procesa valori lipsa
cols_to_fix = ['mileage(km/ltr/kg)', 'engine', 'max_power']
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].mean())

# 2. ELIMINARE ANOMALII
# Folosim metoda statistica IQR (Interquartile Range) pentru a elimina zgomotul din date
# Masinile cu preturi nerealist de mari sau mici ar putea distorsiona grav regulile invatate de modele
Q1 = df['selling_price'].quantile(0.25)
Q3 = df['selling_price'].quantile(0.75)
IQR = Q3 - Q1

# Pastram doar inregistrarile care se afla in intervalul statistic normal
df = df[(df['selling_price'] >= (Q1 - 1.5 * IQR)) & (df['selling_price'] <= (Q3 + 1.5 * IQR))]

# 3. ANALIZA INDICATORILOR
# Cream o copie a datelor pentru a transforma etichetele text in valori numerice
df_encoded = df.copy()
le = LabelEncoder()

# Identificam coloanele de tip text
categorice = df_encoded.select_dtypes(include=['object', 'string']).columns

# Transformam categoriile in numere pentru a permite calculele matematice
for col in categorice:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Calculam si afisam matricea de corelatie Pearson pentru a vedea legaturile liniare cu pretul
print("Corelatie cu Pretul")
print(df_encoded.corr()['selling_price'].sort_values(ascending=False).head(5))

# 4. ANTRENARE SI OPTIMIZARE MODELE
# Separam caracteristicile (X) de variabila tinta / pretul pe care vrem sa il ghicim (y)
X = df_encoded.drop('selling_price', axis=1)
y = df_encoded['selling_price']

# Impartim setul de date: 80% pentru antrenarea modelelor si 20% pentru testarea performantei lor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Functie ajutatoare pentru evaluarea modelelor folosind metrici standard de regresie
def evalueaza(nume, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) # Eroarea medie absoluta in bani
    r2 = r2_score(y_true, y_pred)             # Proportia de varianta explicata de model (0-1)
    print(f"[{nume}] MAE: {mae:.2f}, R2: {r2:.4f}")
    return mae, r2

# Dictionar in care vom salva performantele fiecarui algoritm pentru comparatia finala
models_data = {}

# A1: Linear Regression
# Model simplu, liniar, folosit ca punct de plecare
lr = LinearRegression()
lr.fit(X_train, y_train)
models_data['Linear Regression'] = evalueaza("Linear Regression", y_test, lr.predict(X_test))

# A2: Random Forest
# Model robust bazat pe o multime de arbori de decizie paraleli
print("\nOptimizare Random Forest")
# Folosim GridSearchCV pentru reglarea automata a hiperparametrilor
# Ruleaza o validare incrucisata pentru a gasi combinatia optima de parametri
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [100, 150], 'max_depth': [10, 20]}, cv=3).fit(X_train, y_train)
best_rf = rf_grid.best_estimator_ # Salvam cel mai bun model rezultat din optimizare
models_data['Random Forest'] = evalueaza("Random Forest", y_test, best_rf.predict(X_test))

# A3: Gradient Boosting
# Model avansat de ansamblu care antreneaza arbori secvential, corectand erorile anterioare
print("Optimizare Gradient Boosting")
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05]}, cv=3).fit(X_train, y_train)
best_gb = gb_grid.best_estimator_
models_data['Gradient Boosting'] = evalueaza("Gradient Boosting", y_test, best_gb.predict(X_test))

# 5. ALEGERE MODEL FINAL
print("\nREZULTAT FINAL")
for m, res in models_data.items():
    print(f"{m}: R2 Score = {res[1]:.4f}")

print("\nModelul ales: Random Forest (datorita stabilitatii si scorului R2 ridicat)")

# 6. ANALIZA MODELULUI SI EXPLICABILITATE
# Extragem importanta globala a caracteristicilor calculata de Random Forest
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Caracteristica': feature_names, 'Importanta': importances}).sort_values(by='Importanta', ascending=False)
print("\nImportanta Caracteristicilor (Top 5)")
print(feature_importance_df.head(5))

# Generam predictiile modelului final pe datele de test pentru analiza locala a erorilor
y_pred = best_rf.predict(X_test)
error_analysis = pd.DataFrame({'Actual': y_test, 'Predictie': y_pred})
error_analysis['Eroare'] = abs(error_analysis['Actual'] - error_analysis['Predictie'])

# Identificam indexurile din DataFrame-ul original pentru cele doua cazuri de interes
best_case_idx = error_analysis['Eroare'].idxmin()  # Instanta unde modelul a avut dreptate perfecta
worst_case_idx = error_analysis['Eroare'].idxmax() # Instanta unde modelul a gresit cel mai mult

# Aflam pozitiile matematice corespunzatoare in vectorul de predictii
pos_best = np.where(X_test.index == best_case_idx)[0][0]
pos_worst = np.where(X_test.index == worst_case_idx)[0][0]

print("\nEXPLICABILITATEA MODELULUI")

# Demonstratie de functionare ideala a algoritmului inteligent
print(f"PREDICTIE IDEALA (Eroare minima)")
print(f"Index masina: {best_case_idx}")
print(f"Pret Real: {y_test.loc[best_case_idx]} | Pret Prezis: {y_pred[pos_best]:.2f}")
print("Detalii vehicul:")
# Afisam datele text din tabelul initial pentru a fi intelese cu usurinta de utilizator
print(df.loc[best_case_idx][['name', 'year', 'km_driven', 'fuel', 'max_power']])

# Identificarea limitarilor si a vulnerabilitatilor modelului de AI
print(f"\nEROARE MAXIMA (Limitare a modelului)")
print(f"Index masina: {worst_case_idx}")
print(f"Pret Real: {y_test.loc[worst_case_idx]} | Pret Prezis: {y_pred[pos_worst]:.2f}")
print("Detalii vehicul:")
print(df.loc[worst_case_idx][['name', 'year', 'km_driven', 'fuel', 'max_power']])
