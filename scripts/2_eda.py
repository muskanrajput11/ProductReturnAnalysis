import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load cleaned data
df = pd.read_csv("data/cleaned_data.csv")
print(df['is_returned'].value_counts())


#  Step 2: Basic info check
print("🔹 Shape:", df.shape)
print("🔹 Columns:", df.columns.tolist())
print("🔹 Unique categories:", df['product_category'].nunique())

# Step 3: Return rate by product category
category_return = df.groupby('product_category')['is_returned'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=category_return.values, y=category_return.index, palette="coolwarm")
plt.title("Return Rate by Product Category")
plt.xlabel("Return Rate")
plt.ylabel("Product Category")
plt.tight_layout()

# Step 4: Save plot to reports folder
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/return_rate_by_category.png")
plt.show()

# Step 5: Time trend analysis (monthly return rate)
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df = df.dropna(subset=["order_date"])

df['month'] = df['order_date'].dt.to_period('M')
monthly_return = df.groupby('month')['is_returned'].mean()

plt.figure(figsize=(10, 5))
monthly_return.plot(kind='line', marker='o')
plt.title("Monthly Return Rate Trend")
plt.xlabel("Month")
plt.ylabel("Return Rate")
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/monthly_return_trend.png")
plt.show()

# Step 6: Save summary table
summary = category_return.reset_index()
summary.columns = ['product_category', 'return_rate']

# Save summary in both formats
summary.to_csv("reports/category_return_summary.csv", index=False)
summary.to_excel("reports/category_return_summary.xlsx", index=False)

print("EDA complete. Charts and summary saved in /reports folder.")

