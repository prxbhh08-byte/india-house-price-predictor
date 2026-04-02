# 🏠 India House Price Predictor

A Machine Learning project that predicts house prices in India using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 📌 What This Project Does

- Loads and cleans real India house price data
- Automatically finds the best value of **K** for KNN
- Takes user input (area, bedrooms, locality etc.)
- Predicts house price in **Lakhs**
- Also shows the **expected price range** based on model error

---

## 📂 Project Structure

```
├── ml.py                        # Main Python code
├── india_house_price_data.csv   # Dataset (CSV)
├── requirements.txt             # Required libraries
└── README.md                    # This file
```

---

## 📊 Dataset

File: `india_house_price_data.csv`

| Column | Description |
|--------|-------------|
| `area` | Area of house in sq ft |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `furnished` | Furnished / Unfurnished / Semi |
| `locality` | Location of the house |
| `city` | City (dropped during preprocessing) |
| `price_lakhs` | Price in Lakhs *(target)* |

---

## ⚙️ How to Run

**Step 1 — Clone the repo**
```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
```

**Step 2 — Install libraries**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the code**
```bash
python ml.py
```

**Step 4 — Enter details when asked**
```
Area (sq ft): 1200
Bedrooms: 3
Bathrooms: 2
Furnished (yes/no/semi): yes
Locality: Karol Bagh
```

---

## 📈 Sample Output

```
k=3  → MAE: 18.50
k=4  → MAE: 16.20
k=5  → MAE: 14.80
...
Best k = 5, MAE = 14.80 Lakhs

Predicted Price : 85.40 Lakhs
Model MAE       : ±14.80 Lakhs
Expected Range  : 70.60 – 100.20 Lakhs
```

---

## 🛠️ Libraries Used

- `pandas` — Data loading and processing
- `numpy` — Numerical operations
- `scikit-learn` — KNN model, scaling, evaluation

---

## 👨‍💻 Author

Made by **Prabh**
- GitHub: [@your-username](https://github.com/your-username)

---

## 📝 Notes

- Model uses **KNN Regression** with automatic K selection
- `city` column is dropped (low information)
- `locality` and `furnished` are one-hot encoded
- Scaling is done using **StandardScaler**
