# CKD Prediction Web App

## Overview

The **Chronic Kidney Disease (CKD) Prediction App** is an interactive **Streamlit web application** that predicts whether a patient has CKD based on various health parameters. The app uses a **Decision Tree Classifier** trained on a cleaned CKD dataset.

---

## Features

- User-friendly interface for inputting patient health data  
- Dynamic handling of categorical and numerical features  
- Real-time prediction of CKD status  
- Displays **model performance metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix  

---

## Dataset

The app uses the **CKD dataset** (`kidney_disease.csv`) with the following features:

- Age, Blood Pressure, Specific Gravity, Albumin, Sugar  
- Red Blood Cells, Pus Cell, Pus Cell Clumps, Bacteria  
- Blood Glucose Random, Blood Urea, Serum Creatinine  
- Sodium, Potassium, Hemoglobin, Packed Cell Volume  
- White Blood Cell Count, Red Blood Cell Count  
- Hypertension, Diabetes Mellitus, Coronary Artery Disease  
- Appetite, Pedal Edema, Anemia  

**Target variable**: `classification` (`ckd` = 1, `notckd` = 0)

---

## Technologies Used

- **Python 3.x**  
- **Streamlit** – Interactive web app  
- **Scikit-learn** – Machine learning model (Decision Tree Classifier)  
- **Pandas & NumPy** – Data preprocessing


## Author

 - Azarudeen – Developer | Powered by Streamlit + Scikit-learn
- **LabelEncoder** – Encode categorical variables  

---
