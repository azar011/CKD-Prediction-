import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 📌 Load and Preprocess Dataset
# -------------------------------
df = pd.read_csv("kidney_disease.csv")
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
df.dropna(inplace=True)
df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0})

# 🔁 Label Encoding
label_cols = df.select_dtypes(include='object').columns
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # save encoder for each column

X = df.drop('classification', axis=1)
y = df['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# -------------------------------
# 💡 Feature Full Names Dictionary
# -------------------------------
feature_full_names = {
    'age': 'Age (years)',
    'bp': 'Blood Pressure (mm/Hg)',
    'sg': 'Specific Gravity',
    'al': 'Albumin',
    'su': 'Sugar',
    'rbc': 'Red Blood Cells',
    'pc': 'Pus Cell',
    'pcc': 'Pus Cell Clumps',
    'ba': 'Bacteria',
    'bgr': 'Blood Glucose Random (mg/dl)',
    'bu': 'Blood Urea (mg/dl)',
    'sc': 'Serum Creatinine (mg/dl)',
    'sod': 'Sodium (mEq/L)',
    'pot': 'Potassium (mEq/L)',
    'hemo': 'Hemoglobin (gms)',
    'pcv': 'Packed Cell Volume',
    'wc': 'White Blood Cell Count (cells/cumm)',
    'rc': 'Red Blood Cell Count (millions/cmm)',
    'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus',
    'cad': 'Coronary Artery Disease',
    'appet': 'Appetite',
    'pe': 'Pedal Edema',
    'ane': 'Anemia'
}

# -------------------------------
# 🎨 Streamlit UI Design
# -------------------------------
st.set_page_config(page_title="CKD Predictor", layout="wide")
st.title("🩺 Chronic Kidney Disease (CKD) Prediction App")
st.markdown("Enter the patient details below to predict whether they have **CKD or Not**.")

st.markdown("---")
st.subheader("🔽 Patient Input Details")

# Create columns layout
cols = st.columns(3)
user_input = {}

# Build dynamic form
for i, feature in enumerate(X.columns):
    label = feature_full_names.get(feature, feature.capitalize())
    col = cols[i % 3]
    if feature in label_cols:
        options = le_dict[feature].classes_
        choice = col.selectbox(f"{label}:", options)
        user_input[feature] = le_dict[feature].transform([choice])[0]
    else:
        min_val = int(df[feature].min()) if feature == "age" else float(df[feature].min())
        max_val = int(df[feature].max()) if feature == "age" else float(df[feature].max())
        mean_val = int(df[feature].mean()) if feature == "age" else float(df[feature].mean())
        step = 1 if feature == "age" else (0.1 if df[feature].dtype == 'float64' else 1.0)
        user_input[feature] = col.slider(
            f"{label}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=step,
        )

st.markdown("---")

# --------------- Predict ---------------
if st.button("🔍 Predict CKD Status"):
    user_df = pd.DataFrame([user_input])
    prediction = model.predict(user_df)[0]
    result = "🟥 CKD Detected" if prediction == 1 else "🟩 No CKD (Healthy)"
    st.success(f"✅ **Prediction:** {result}")

# --------------- Show Metrics ---------------
with st.expander("📊 Model Performance Metrics"):
    y_pred = model.predict(X_test)
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.4f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.4f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.4f}")
    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))

st.markdown("---")
st.caption("🔗 Developed by Azarudeen | Powered by Streamlit + Scikit-learn")
