from flask import Flask, render_template, request
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

app = Flask(__name__)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target,

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly']
}

# StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train SVM with GridSearch
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Take 5 main features (to keep form simple)
        features = [float(request.form.get(f)) for f in [
            "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"
        ]]
        # final_features = np.array(features).reshape(1, -1)

        # Scale input using same scaler (only 5 features, so slice)
        all_features = np.zeros((1, X.shape[1]))  # create empty feature array
        all_features[0, :5] = features  # fill first 5 features
        all_features_scaled = scaler.transform(all_features)

        prediction = best_model.predict(all_features_scaled)[0]
        probability = best_model.predict_proba(all_features_scaled)[0][prediction]

        result = "⚠️ High Risk of Breast Cancer" if prediction == 0 else "✅ Low Risk of Breast Cancer"

        return render_template('result.html', result=result, probability=f"{probability*100:.2f}%")
    except Exception as e:
        return render_template('result.html', result="Error: " + str(e), probability="N/A")

if __name__ == "__main__":
    app.run(debug=True)
