# 🩺 Breast Cancer Prediction Web App (SVM + Flask)

## 📌 Overview

This project is a **Flask-based Machine Learning web application** that predicts the likelihood of breast cancer using a **Support Vector Machine (SVM)** model.

The model is trained using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.  
It applies:

- Data preprocessing (Standardization)
- Stratified cross-validation
- Hyperparameter tuning using GridSearchCV
- Probability-based predictions

The web interface allows users to input selected medical features and receive a risk prediction with confidence probability.

---

## 🏗️ Tech Stack

- **Backend Framework:** Flask  
- **Machine Learning:** Scikit-learn  
- **Model:** Support Vector Machine (SVC)  
- **Hyperparameter Tuning:** GridSearchCV  
- **Cross Validation:** StratifiedKFold  
- **Data Processing:** StandardScaler  
- **Numerical Computing:** NumPy  

---

## 📊 Dataset

Dataset used:

```
sklearn.datasets.load_breast_cancer()
```

- Total Features: 30
- Target Classes:
  - 0 → Malignant
  - 1 → Benign

For simplicity, the web form uses **5 primary features**:

1. Mean Radius  
2. Mean Texture  
3. Mean Perimeter  
4. Mean Area  
5. Mean Smoothness  

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Loading

```python
data = load_breast_cancer()
X, y = data.data, data.target
```

---

### 2️⃣ Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

- 80% Training Data  
- 20% Testing Data  
- Fixed random state for reproducibility  

---

### 3️⃣ Feature Standardization

```python
scaler = StandardScaler()
```

Why scaling?

- SVM is distance-based
- Ensures equal feature contribution
- Prevents dominance of large-scale values

---

### 4️⃣ Hyperparameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly']
}
```

Parameters Explained:

- **C** → Regularization strength  
- **gamma** → Kernel coefficient  
- **kernel** → Decision boundary function  

---

### 5️⃣ Cross Validation

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- Maintains class distribution
- Improves generalization
- Prevents bias in small datasets

---

### 6️⃣ Grid Search Training

```python
GridSearchCV(SVC(probability=True), param_grid, cv=cv)
```

- Searches best hyperparameters
- Uses cross-validation
- Selects best performing model

Final model:

```python
best_model = grid_search.best_estimator_
```

---



### Process:

1. Collect user input (5 features)
2. Convert input to float
3. Create full 30-feature array (remaining values set to 0)
4. Apply previously fitted scaler
5. Predict class using trained SVM
6. Get prediction probability
7. Display result in `result.html`

---

## 🧠 Prediction Logic

```python
prediction = best_model.predict(all_features_scaled)[0]
probability = best_model.predict_proba(all_features_scaled)[0][prediction]
```

### Output Labels:

- `0` → ⚠️ High Risk of Breast Cancer (Malignant)
- `1` → ✅ Low Risk of Breast Cancer (Benign)

Probability is shown as a percentage.

---

## 📂 Project Structure

```
project/
│
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
└── README.md
```

---

## ▶️ Installation & Run

### Install Dependencies

```bash
pip install flask scikit-learn numpy
```

### Run Application

```bash
python app.py
```

Access in browser:

```
http://127.0.0.1:5000/
```

---

## 🛡️ Error Handling

The application includes a try-except block:

```python
except Exception as e:
```

If an error occurs:
- Displays error message
- Prevents server crash

---

## 🎯 Key Advantages

- Automated hyperparameter tuning  
- Stratified cross-validation  
- Probability-based prediction  
- Simple and user-friendly interface  
- Clean ML pipeline  

---



## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.

It is **not a medical diagnostic tool** and must not be used for real-world medical decisions. Always consult a qualified healthcare professional for medical advice.
