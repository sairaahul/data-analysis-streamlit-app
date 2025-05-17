
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from io import BytesIO

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")
st.title("📊 Smart Data Analyzer with Insights")

# --- ADVANCED FILTERING ---
def add_filtering_ui(df):
    st.sidebar.subheader("🔍 Advanced Filtering")
    filtered_df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        options = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
        filtered_df = filtered_df[filtered_df[col].isin(selected)]
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        range_val = st.sidebar.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[df[col].between(*range_val)]
    return filtered_df

# --- EXPORT BUTTON ---
def download_button(df, label="Download CSV"):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(label=label, data=buffer, file_name="exported_data.csv", mime="text/csv")

# --- AUTO-VISUALIZATION ---
def auto_visualize(df):
    st.subheader("📈 Smart Visualizations")
    for col in df.select_dtypes(include=['object']).columns[:2]:
        if df[col].nunique() <= 10:
            st.markdown(f"**Distribution of {col}**")
            fig, ax = plt.subplots()
            df[col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)

    for col in df.select_dtypes(include=[np.number]).columns[:3]:
        st.line_chart(df[col], height=250)
        st.caption(f"📉 Trend line of **{col}**")

    for num_col in df.select_dtypes(include=[np.number]).columns[:2]:
        for cat_col in df.select_dtypes(include=['object']).columns[:1]:
            st.markdown(f"**Bar plot of {num_col} by {cat_col}**")
            fig, ax = plt.subplots()
            sns.barplot(x=cat_col, y=num_col, data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

# --- MODEL PREDICTION ---
def model_prediction_ui(df):
    st.markdown("<div class='section-header'>🤖 Model Prediction</div>", unsafe_allow_html=True)
    target = st.selectbox("🎯 Select target column", df.columns)
    task_type = "classification" if df[target].nunique() < 10 else "regression"
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
    y = df[target].loc[X.index]

    if X.empty or y.isnull().any():
        st.warning("⚠️ Insufficient numeric data or missing target values for modeling.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if task_type == "classification":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.success(f"✅ Accuracy: {accuracy_score(y_test, preds):.2f}")
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.success(f"✅ RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}")

    df[f"Predicted_{target}"] = model.predict(X)
    st.dataframe(df[[target, f"Predicted_{target}"]].head())
    download_button(df, label="📥 Download Predictions")

# --- INSIGHT REPORT ---
def generate_interpretation(df):
    st.markdown("<div class='section-header'>📋 Summary & Business Insights</div>", unsafe_allow_html=True)
    st.write(f"✅ The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns** after cleaning.")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    st.markdown("### 🔍 Key Insights")
    if len(num_cols) > 0:
        top_var = df[num_cols].std().sort_values(ascending=False).index[0]
        st.markdown(f"- **{top_var}** shows the most variability and could be a driver of performance or cost.")
    if len(cat_cols) > 0:
        top_cat = cat_cols[0]
        st.markdown(f"- **{top_cat}** has {df[top_cat].nunique()} unique categories; consider segmenting strategies based on this.")

    st.markdown("- Consider running predictive models to better understand future trends and improve decision-making.")
    st.markdown("- Ensure any outliers are investigated, especially in high-variance numeric fields.")

# --- MAIN EXECUTION ---
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    df.dropna(inplace=True)
    st.success(f"✅ Null values removed. Cleaned data shape: {df.shape}")
    df = add_filtering_ui(df)

    st.subheader("👀 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📌 Summary Statistics")
    st.dataframe(df.describe(include='all').T)

    auto_visualize(df)
    generate_interpretation(df)
    model_prediction_ui(df)
    download_button(df, label="📥 Download Final Dataset")
