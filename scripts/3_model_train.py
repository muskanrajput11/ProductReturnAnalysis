import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")
# Check target distribution
print(df["is_returned"].value_counts())

# Impute missing values instead of dropping everything
cat_cols = ["product_category", "return_reason", "user_gender", "payment_method", "shipping_method"]
num_cols = ["product_price", "order_quantity", "user_age", "discount_applied"]

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=["is_returned"])

# Check again after dropping
print("After dropna:")
print(df["is_returned"].value_counts())  # ✅ Debug

# Define features and target
features = [
    "product_category", "return_reason", "product_price", "order_quantity",
    "user_age", "user_gender", "payment_method", "shipping_method", "discount_applied"
]
X = df[features].copy()  # ✅ Ensure no chained assignment issues
y = df["is_returned"]

# Check class balance
if len(y.unique()) < 2:
    print("❌ Not enough classes in target variable.")
    print("Class distribution:\n", y.value_counts())
    exit()

# Encode categorical features
cat_features = ["product_category", "return_reason", "user_gender", "payment_method", "shipping_method"]
encoders = {}
for col in cat_features:
    encoder = LabelEncoder()
    X.loc[:, col] = encoder.fit_transform(X[col])
    encoders[col] = encoder

# Scale numeric features
numeric_features = ["product_price", "order_quantity", "user_age", "discount_applied"]
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/return_predictor.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl") 

for col, encoder in encoders.items():
    joblib.dump(encoder, f"models/{col}_encoder.pkl")

print("✅ Model and encoders saved successfully.")



