import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# ── 1. DATA LOAD ──────────────────────────────
df = pd.read_csv('india_house_price_data.csv')
df = df.drop('city', axis=1)
df = pd.get_dummies(df, columns=['locality', 'furnished'])

# ── 2. FEATURES & TARGET ──────────────────────
X = df.drop('price_lakhs', axis=1).astype(float)
y = df['price_lakhs']

# ── 3. SPLIT ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. SCALING ────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 5. BEST K DHUNDHO ─────────────────────────
best_k   = 3
best_mae = float('inf')

for k in range(3, 21):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"k={k} → MAE: {mae:.2f}")
    if mae < best_mae:
        best_mae = mae
        best_k   = k

print(f"\nBest k = {best_k}, MAE = {best_mae:.2f} Lakhs\n")

# ── 6. FINAL MODEL ────────────────────────────
model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(X_train, y_train)

# ── 7. USER INPUT ─────────────────────────────
area      = float(input("Area (sq ft): "))
bedrooms  = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
furnished = input("Furnished (yes/no/semi): ").strip().lower()
locality  = input("Locality: ").strip()

# ── 8. INPUT ROW BANANA ───────────────────────
row = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)

row['area']      = area
row['bedrooms']  = bedrooms
row['bathrooms'] = bathrooms

# Furnished set karo
f_col = f"furnished_{furnished}"
if f_col in row.columns:
    row[f_col] = 1.0
else:
    print("Furnished type galat hai!")
    print("Valid options:", [c.replace("furnished_","") for c in X.columns if "furnished_" in c])

# Locality set karo
l_col = f"locality_{locality}"
if l_col in row.columns:
    row[l_col] = 1.0
else:
    print("Locality nahi mili!")
    print("Kuch valid localities:", [c.replace("locality_","") for c in X.columns if "locality_" in c][:5])

# ── 9. PREDICT ────────────────────────────────
row_scaled = scaler.transform(row.astype(float).values)
price      = model.predict(row_scaled)[0]

print(f"\nPredicted Price : {price:.2f} Lakhs")
print(f"Model MAE       : ±{best_mae:.2f} Lakhs")
print(f"Expected Range  : {max(0, price - best_mae):.2f} – {price + best_mae:.2f} Lakhs")
