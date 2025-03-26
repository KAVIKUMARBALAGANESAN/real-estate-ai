import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset (Ensure correct separator)
df = pd.read_csv("data/real_estate_data.csv", sep="\t")

# Rename columns if necessary
df.rename(columns={'locality_name': 'location', 'total_area': 'size', 'rooms': 'bedrooms', 'kitchen_area': 'kitchen'}, inplace=True)

# ✅ Check for missing values
#print("Missing values before handling:\n", df.isnull().sum())

# ✅ Fill missing numeric values with the median
numeric_cols = ['size', 'bedrooms', 'kitchen', 'last_price']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ✅ Fill missing categorical values with the most common value (mode)
df['location'] = df['location'].fillna(df['location'].mode()[0])

# ✅ Check again after handling NaNs
#print("Missing values after handling:\n", df.isnull().sum())

# Features & Target
X = df[['location', 'size', 'bedrooms', 'kitchen']]
y = df['last_price']

# Convert categorical data (location)
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved as model.pkl")
