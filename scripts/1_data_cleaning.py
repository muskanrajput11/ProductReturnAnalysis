import pandas as pd

# Step 1: Raw data load karo
df = pd.read_csv("data/ecommerce_returns_synthetic_data.csv")

# Step 2: Columns ko lowercase aur underscores mein convert karo
df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

df['is_returned'] = df['return_date'].notnull().astype(int)

# Step 3: Data types check karo
print(df.dtypes)
print(df.head())

# Step 4: Missing values dekh lo
print("Missing values:\n", df.isnull().sum())

#  Step 5: Missing values ko handle karo
# Example: fillna ya dropna based on columns
df = df.dropna(subset=["product_id", "order_date"])  # important columns
df['product_category'].fillna("Unknown", inplace=True)

# Step 6: Date format correct karo
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

#  Step 7: Export cleaned data
df.to_csv("data/cleaned_data.csv", index=False)
print("Data cleaned and saved to data/cleaned_data.csv")


