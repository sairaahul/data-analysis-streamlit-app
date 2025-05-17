
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt

# --- ADVANCED FILTERING ---
def add_filtering_ui(df):
    st.sidebar.subheader("🔍 Advanced Filtering")
    filtered_df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"Filter {col}", options, default=options)
        filtered_df = filtered_df[filtered_df[col].isin(selected)]
    for col in df.select_dtypes(include=[np.number]).columns:
        min_val, max_val = float(df[col].min()), float(df[col].max())
        range_val = st.sidebar.slider(f"Range for {col}", min_val, max_val, (min_val, max_val))
        filtered_df = filtered_df[filtered_df[col].between(*range_val)]
    return filtered_df

# --- EXPORT BUTTON ---
def download_button(df, label="Download CSV"):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(label=label, data=buffer, file_name="exported_data.csv", mime="text/csv")

# --- MODEL PREDICTION ---
def model_prediction_ui(df):
    st.markdown("<div class='section-header'>🤖 Model Prediction</div>", unsafe_allow_html=True)
    target = st.selectbox("Select target column", df.columns)
    task_type = "classification" if df[target].nunique() < 10 else "regression"
    X = df.drop(columns=[target]).select_dtypes(include=[np.number]).dropna()
    y = df[target].loc[X.index]
    if X.empty or y.isnull().any():
        st.warning("Insufficient numeric data or missing target values for modeling.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if task_type == "classification":
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.success(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
    else:
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        st.success(f"RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}")

    full_preds = model.predict(X)
    df[f"Predicted_{target}"] = full_preds
    st.dataframe(df[[target, f"Predicted_{target}"]].head())
    download_button(df, label="Download Predictions")

# --- INTERPRETATION BASED ON VISUALS ---
def generate_interpretation(df):
    st.markdown("<div class='section-header'>📋 Summary Interpretation</div>", unsafe_allow_html=True)
    st.write(f"The dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    num_cols = df.select_dtypes(include=[np.number])
    if not num_cols.empty:
        st.write("### Key Numeric Insights")
        for col in num_cols.columns:
            mean_val = num_cols[col].mean()
            median_val = num_cols[col].median()
            std_dev = num_cols[col].std()
            st.markdown(f"- **{col}**: Mean = `{mean_val:.2f}`, Median = `{median_val:.2f}`, Std Dev = `{std_dev:.2f}`")
            if std_dev > 0.3 * mean_val:
                st.markdown(f"  ⚠️ *High variance detected in {col}*")
            if mean_val > median_val:
                st.markdown(f"  🔺 *Right-skewed distribution*")
            elif mean_val < median_val:
                st.markdown(f"  🔻 *Left-skewed distribution*")

# --- MAIN APP ---
if __name__ == "__main__":
    st.set_page_config(page_title="End-to-End Data Analysis", layout="wide")
    st.title("📊 End-to-End Data Analyzer")
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.info("🧹 Dropping rows with any null values...")
        df.dropna(inplace=True)
        st.success(f"✅ Cleaned dataset. Shape: {df.shape}")

        df = add_filtering_ui(df)

        st.subheader("👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📌 Summary Statistics")
        st.markdown("### 🧮 Numeric Columns")
        st.dataframe(df.select_dtypes(include=[np.number]).describe().T)

        st.markdown("### 🏷️ Categorical Columns")
        cat_desc = df.select_dtypes(include=['object']).describe().T
        if not cat_desc.empty:
            st.dataframe(cat_desc)

        st.subheader("📈 Visualizations")
        num_cols = df.select_dtypes(include=[np.number]).columns[:3]
        for col in num_cols:
            st.markdown(f"#### 📌 {col}")
            st.line_chart(df[col], height=250)
            st.caption(f"Trend line of {col}")

            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            sns.histplot(df[col], bins=30, ax=ax[0], kde=True)
            ax[0].set_title(f"Histogram: {col}")
            sns.boxplot(x=df[col], ax=ax[1])
            ax[1].set_title(f"Boxplot: {col}")
            st.pyplot(fig)

        if len(df.select_dtypes(include=[np.number]).columns) >= 2:
            st.markdown("### 📊 Correlation Heatmap")
            corr = df.select_dtypes(include=[np.number]).corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

        st.markdown("### 🏷️ Categorical Bar Charts")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols[:2]:
            st.markdown(f"#### 📌 {col} Distribution")
            top_categories = df[col].value_counts().nlargest(10)
            st.bar_chart(top_categories)

        st.markdown("### 🕒 Time Series Trend (if applicable)")
        datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        if len(datetime_cols) > 0:
            date_col = st.selectbox("Select date column for time trend", datetime_cols)
            metric = st.selectbox("Select numeric metric to plot over time", df.select_dtypes(include=[np.number]).columns)
            df_time = df.sort_values(by=date_col)
            df_time = df_time[[date_col, metric]].groupby(date_col).mean().reset_index()
            st.line_chart(df_time.set_index(date_col))

        generate_interpretation(df)
        model_prediction_ui(df)
        download_button(df, label="Download Final Dataset")

