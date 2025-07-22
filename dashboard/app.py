import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Page Configuration with custom theme and font
st.set_page_config(
    page_title="Product Return Dashboard",
    layout="wide",
)

# Custom CSS styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9f9fb;
    }
    h1 {
        color: #3E64FF;
    }
    .stButton > button {
        background-color: #3E64FF;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .stDownloadButton > button {
        background-color: #12B886;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
    }
    .stMetric > div {
        background-color: #eef2ff;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
df = pd.read_csv("data/cleaned_data.csv")
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
df.dropna(subset=['order_date'], inplace=True)
df['month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

# Sidebar Filters
st.sidebar.header(" Filters")

categories = df['product_category'].dropna().unique().tolist()
selected_category = st.sidebar.selectbox("Select Product Category", categories)

st.sidebar.subheader("Select Date Range")
min_date = df['order_date'].min().date()
max_date = df['order_date'].max().date()
start_date, end_date = st.sidebar.date_input("Order Date Range", [min_date, max_date])

# Filter Data
mask = (df['order_date'] >= pd.to_datetime(start_date)) & (df['order_date'] <= pd.to_datetime(end_date))
df = df[mask]
filtered_df = df[df['product_category'] == selected_category]

# KPIs
return_rate = filtered_df['is_returned'].mean()

# Monthly Return Trend
monthly_return = (
    filtered_df.groupby('month')['is_returned']
    .mean()
    .sort_index()
    .reset_index()
)

# Title
st.markdown("<h1 style='text-align:center;'>Product Return Pattern Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Selected Category")
    st.success(selected_category)
with col2:
    st.markdown("###  Return Rate")
    st.metric(label="", value=f"{return_rate:.2%}")

# Line Chart
st.markdown("###  Monthly Return Trend")
fig = px.line(
    monthly_return,
    x='month',
    y='is_returned',
    title='Monthly Return Rate Trend',
    markers=True,
    template="plotly_white"
)
fig.update_traces(line_color='#3E64FF')
fig.update_layout(
    yaxis_tickformat='.2%',
    title_font=dict(size=20, color='#3E64FF'),
    title_x=0.2
)
st.plotly_chart(fig, use_container_width=True)

# Category Summary
st.markdown("###  Return Rate by Category")
category_summary = (
    df.groupby('product_category')['is_returned']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={'is_returned': 'return_rate'})
)
st.dataframe(category_summary.style.format({'return_rate': '{:.2%}'}), use_container_width=True)

# Load Model + Encoders
model = joblib.load("models/return_predictor.pkl")
scaler = joblib.load("models/feature_scaler.pkl")
cat_encoder = {
    'product_category': joblib.load("models/product_category_encoder.pkl"),
    'return_reason': joblib.load("models/return_reason_encoder.pkl"),
    'user_gender': joblib.load("models/user_gender_encoder.pkl"),
    'payment_method': joblib.load("models/payment_method_encoder.pkl"),
    'shipping_method': joblib.load("models/shipping_method_encoder.pkl"),
}

#  Prediction
st.markdown("###  Predict Return Probability")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        pcat = st.selectbox("Product Category", df['product_category'].unique())
        reason = st.selectbox("Return Reason", df['return_reason'].dropna().unique())
        gender = st.selectbox("User Gender", df['user_gender'].unique())
        shipping = st.selectbox("Shipping Method", df['shipping_method'].unique())
    with col2:
        price = st.number_input("Product Price", min_value=0.0, value=500.0)
        qty = st.number_input("Order Quantity", min_value=1, value=1)
        age = st.slider("User Age", min_value=10, max_value=100, value=30)
        payment = st.selectbox("Payment Method", df['payment_method'].unique())
        discount = st.slider("Discount Applied (%)", 0.0, 100.0, 10.0)

    submitted = st.form_submit_button(" Predict")

    if submitted:
        try:
            input_data = {
                'product_category': cat_encoder['product_category'].transform([pcat])[0],
                'return_reason': cat_encoder['return_reason'].transform([reason])[0],
                'user_gender': cat_encoder['user_gender'].transform([gender])[0],
                'payment_method': cat_encoder['payment_method'].transform([payment])[0],
                'shipping_method': cat_encoder['shipping_method'].transform([shipping])[0],
                'product_price': price,
                'order_quantity': qty,
                'user_age': age,
                'discount_applied': discount,
            }

            input_df = pd.DataFrame([input_data])

            # Scale numeric
            num_cols = ["product_price", "order_quantity", "user_age", "discount_applied"]
            scaled_array = scaler.transform(input_df[num_cols])
            scaled_df = pd.DataFrame(scaled_array, columns=num_cols)
            input_df.update(scaled_df)

            # Reorder features
            final_columns = model.feature_names_in_
            input_df = input_df[final_columns]

            # Predict
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f" Likely to be returned ({prob:.2%} confidence)")
            else:
                st.success(f"Unlikely to be returned ({prob:.2%} confidence)")

            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []

            st.session_state.prediction_history.append({
                "Product Category": pcat,
                "Return Reason": reason,
                "Gender": gender,
                "Shipping": shipping,
                "Price": price,
                "Qty": qty,
                "Age": age,
                "Payment": payment,
                "Discount": discount,
                "Prediction": "Returned" if prediction == 1 else "Not Returned",
                "Confidence": f"{prob:.2%}"
            })

        except Exception as e:
            st.warning(f"Prediction failed: {e}")

# Prediction History Section
if "prediction_history" in st.session_state and st.session_state.prediction_history:
    st.markdown("###  Prediction History")
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df, use_container_width=True)

    history_csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download Prediction History",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🧹 Clear Prediction History"):
        st.session_state.prediction_history.clear()
        st.success("Prediction history cleared.")

# Report Download for Category Summary
st.markdown("###  Download Category Report")
csv = category_summary.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇ Download CSV File",
    data=csv,
    file_name='category_return_summary.csv',
    mime='text/csv',
    help="Download category-wise return rate data"
)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import joblib

# # Page Configuration
# st.set_page_config(page_title="Product Return Dashboard", layout="wide")

# # Load Data
# df = pd.read_csv("data/cleaned_data.csv")
# df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
# df.dropna(subset=['order_date'], inplace=True)
# df['month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

# # Sidebar Filters
# st.sidebar.header("Filters")

# categories = df['product_category'].dropna().unique().tolist()
# selected_category = st.sidebar.selectbox("Select Product Category", categories)

# st.sidebar.subheader("Select Date Range")
# min_date = df['order_date'].min().date()
# max_date = df['order_date'].max().date()
# start_date, end_date = st.sidebar.date_input("Order Date Range", [min_date, max_date])

# # Filter Data
# mask = (df['order_date'] >= pd.to_datetime(start_date)) & (df['order_date'] <= pd.to_datetime(end_date))
# df = df[mask]
# filtered_df = df[df['product_category'] == selected_category]

# # KPIs
# return_rate = filtered_df['is_returned'].mean()

# # Monthly Return Trend
# monthly_return = (
#     filtered_df.groupby('month')['is_returned']
#     .mean()
#     .sort_index()
#     .reset_index()
# )

# # Title
# st.markdown("<h1 style='text-align:center;'>Product Return Pattern Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("---")

# col1, col2 = st.columns(2)
# with col1:
#     st.markdown("### Selected Category")
#     st.success(selected_category)
# with col2:
#     st.markdown("### Return Rate")
#     st.metric(label="", value=f"{return_rate:.2%}")

# # Line Chart
# st.markdown("### Monthly Return Trend")
# fig = px.line(
#     monthly_return,
#     x='month',
#     y='is_returned',
#     title='Monthly Return Rate Trend',
#     markers=True,
#     template="plotly_white"
# )
# fig.update_traces(line_color='blue')
# fig.update_layout(yaxis_tickformat='.2%', title_font=dict(size=20), title_x=0.2)
# st.plotly_chart(fig, use_container_width=True)

# # Category Summary
# st.markdown("### Return Rate by Category")
# category_summary = (
#     df.groupby('product_category')['is_returned']
#     .mean()
#     .sort_values(ascending=False)
#     .reset_index()
#     .rename(columns={'is_returned': 'return_rate'})
# )
# st.dataframe(category_summary.style.format({'return_rate': '{:.2%}'}), use_container_width=True)

# # Load Model + Encoders
# model = joblib.load("models/return_predictor.pkl")
# scaler = joblib.load("models/feature_scaler.pkl")
# cat_encoder = {
#     'product_category': joblib.load("models/product_category_encoder.pkl"),
#     'return_reason': joblib.load("models/return_reason_encoder.pkl"),
#     'user_gender': joblib.load("models/user_gender_encoder.pkl"),
#     'payment_method': joblib.load("models/payment_method_encoder.pkl"),
#     'shipping_method': joblib.load("models/shipping_method_encoder.pkl"),
# }

# #  Prediction
# st.markdown("###  Predict Return Probability")
# with st.form("prediction_form"):
#     col1, col2 = st.columns(2)
#     with col1:
#         pcat = st.selectbox("Product Category", df['product_category'].unique())
#         reason = st.selectbox("Return Reason", df['return_reason'].dropna().unique())
#         gender = st.selectbox("User Gender", df['user_gender'].unique())
#         shipping = st.selectbox("Shipping Method", df['shipping_method'].unique())
#     with col2:
#         price = st.number_input("Product Price", min_value=0.0, value=500.0)
#         qty = st.number_input("Order Quantity", min_value=1, value=1)
#         age = st.slider("User Age", min_value=10, max_value=100, value=30)
#         payment = st.selectbox("Payment Method", df['payment_method'].unique())
#         discount = st.slider("Discount Applied (%)", 0.0, 100.0, 10.0)

#     submitted = st.form_submit_button("Predict")

#     if submitted:
#         try:
#             input_data = {
#                 'product_category': cat_encoder['product_category'].transform([pcat])[0],
#                 'return_reason': cat_encoder['return_reason'].transform([reason])[0],
#                 'user_gender': cat_encoder['user_gender'].transform([gender])[0],
#                 'payment_method': cat_encoder['payment_method'].transform([payment])[0],
#                 'shipping_method': cat_encoder['shipping_method'].transform([shipping])[0],
#                 'product_price': price,
#                 'order_quantity': qty,
#                 'user_age': age,
#                 'discount_applied': discount,
#             }

#             input_df = pd.DataFrame([input_data])

#             # Scale numeric
#             num_cols = ["product_price", "order_quantity", "user_age", "discount_applied"]
#             scaled_array = scaler.transform(input_df[num_cols])
#             scaled_df = pd.DataFrame(scaled_array, columns=num_cols)
#             input_df.update(scaled_df)

#             # Reorder features
#             final_columns = model.feature_names_in_
#             input_df = input_df[final_columns]

#             # Predict
#             prediction = model.predict(input_df)[0]
#             prob = model.predict_proba(input_df)[0][1]

#             if prediction == 1:
#                 st.error(f"Likely to be returned ({prob:.2%} confidence)")
#             else:
#                 st.success(f"Unlikely to be returned ({prob:.2%} confidence)")

#             #  Save to prediction history
#             if "prediction_history" not in st.session_state:
#                 st.session_state.prediction_history = []

#             st.session_state.prediction_history.append({
#                 "Product Category": pcat,
#                 "Return Reason": reason,
#                 "Gender": gender,
#                 "Shipping": shipping,
#                 "Price": price,
#                 "Qty": qty,
#                 "Age": age,
#                 "Payment": payment,
#                 "Discount": discount,
#                 "Prediction": "Returned" if prediction == 1 else "Not Returned",
#                 "Confidence": f"{prob:.2%}"
#             })

#         except Exception as e:
#             st.warning(f" Prediction failed: {e}")

# #  Prediction History Section
# if "prediction_history" in st.session_state and st.session_state.prediction_history:
#     st.markdown("###  Prediction History")
#     history_df = pd.DataFrame(st.session_state.prediction_history)
#     st.dataframe(history_df, use_container_width=True)

#     # Download button
#     history_csv = history_df.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="⬇ Download Prediction History",
#         data=history_csv,
#         file_name="prediction_history.csv",
#         mime="text/csv"
#     )

#     # Clear button
#     if st.button("🧹 Clear Prediction History"):
#         st.session_state.prediction_history.clear()
#         st.success("Prediction history cleared.")

# #  Report Download for Category Summary
# st.markdown("### Download Category Report")
# csv = category_summary.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="⬇ Download CSV File",
#     data=csv,
#     file_name='category_return_summary.csv',
#     mime='text/csv',
#     help="Download category-wise return rate data"
# )



