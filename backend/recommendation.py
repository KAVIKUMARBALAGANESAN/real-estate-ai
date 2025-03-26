import pandas as pd

df = pd.read_csv("data/real_estate_data.csv")

def recommend_properties(location, budget):
    filtered = df[(df['location'] == location) & (df['price'] <= budget)]
    return filtered[['property_name', 'price', 'size', 'bedrooms', 'bathrooms']].to_dict(orient='records')
