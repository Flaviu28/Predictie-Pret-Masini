import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# 1. Incarcare si Preprocesare
df = pd.read_csv('cardekho.csv')
if 'seats' in df.columns: df = df.drop(columns=['seats'])

# Curatare max_power, engine si mileage
df['max_power'] = df['max_power'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
df['engine'] = df['engine'].astype(str).str.extract(r'(\d+)').astype(float)
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# Umplem valorile lipsa cu media
df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].fillna(df['mileage(km/ltr/kg)'].mean())
df['engine'] = df['engine'].fillna(df['engine'].mean())
df['max_power'] = df['max_power'].fillna(df['max_power'].mean())

# Selectam coloanele categorice (text)
categorice = df.select_dtypes(include=['object', 'string']).columns

# Transformam categoriile in numere pentru calcule matematice
df_encoded = df.copy()
le = LabelEncoder()
for col in categorice:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# 1. CORELAȚII (Pearson)
numeric_df = df.select_dtypes(include=[np.number])
cor_pret = numeric_df.corr()['selling_price'].sort_values(ascending=False)
print("Corelatii cu Pretul de Vanzare")
print(cor_pret)

# 2. INFORMATION QUANTITY (Mutual Information)
X = df_encoded.drop('selling_price', axis=1)
y = df_encoded['selling_price']
mi_scores = mutual_info_regression(X, y)
mi_results = pd.Series(mi_scores, name="Info Quantity", index=X.columns).sort_values(ascending=False)
print("\nCantitatea de Informatie (Mutual Info)")
print(mi_results)

# 3. GINI INDEX (Concept de puritate)
# "Impartim" pretul in 2 categorii: ieftin si scump, pentru a exemplifica cum functioneaza indicatorul.
pret_mediu = df['selling_price'].median()
df_gini = df.copy()
df_gini['pret_cat'] = np.where(df_gini['selling_price'] > pret_mediu, 'Scump', 'Ieftin')

def calculeaza_gini(data, coloana):
    proportii = data[coloana].value_counts(normalize=True)
    gini = 1 - sum(proportii**2)
    return gini

gini_total = calculeaza_gini(df_gini, 'pret_cat')
print(f"\nGini Index Total (Impuritatea preturilor): {gini_total:.4f}")

# VIZUALIZARE
plt.figure(figsize=(10, 6))
mi_results.plot(kind='bar', color='skyblue')
plt.title('Cat de multa informatie aduce fiecare coloana pentru pret')
plt.ylabel('Scor Informatie')
plt.show()