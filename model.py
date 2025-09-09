import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')


def train_crop_recommendation_model():
    """Train and save the crop recommendation model."""
    print("Loading dataset...")
    crop = pd.read_csv("Crop_recommendation.csv")

    # Check for missing values
    if crop.isnull().sum().sum() > 0:
        print(f"Found {crop.isnull().sum().sum()} missing values. Cleaning data...")
        crop = crop.dropna()

    # Create label encoding
    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
        'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
        'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
        'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
        'chickpea': 21, 'coffee': 22
    }

    inverse_crop_dict = {v: k for k, v in crop_dict.items()}
    crop['label'] = crop['label'].map(crop_dict)

    # Split features and target
    X = crop.drop('label', axis=1)
    y = crop['label']

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Applying feature scaling...")
    minmax = MinMaxScaler()
    X_train_minmax = minmax.fit_transform(X_train)
    X_test_minmax = minmax.transform(X_test)

    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train_minmax)
    X_test_scaled = std_scaler.transform(X_test_minmax)

    print("Training Random Forest model with cross-validation...")
    base_model = RandomForestClassifier(random_state=42)
    cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=5)
    print(f"Cross Validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")

    print("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nFeature Importances:")
    feature_names = X.columns
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i in range(len(feature_names)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    print("\nSaving model and preprocessing objects...")
    joblib.dump(best_model, 'crop_recommendation_model.pkl')
    joblib.dump(minmax, 'minmax_scaler.pkl')
    joblib.dump(std_scaler, 'standard_scaler.pkl')
    joblib.dump(inverse_crop_dict, 'crop_labels.pkl')

    print("Model training completed successfully!")
    return best_model, minmax, std_scaler, inverse_crop_dict


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict the recommended crop based on soil and climate parameters.
    Returns:
        Tuple(str, float): Recommended crop name, Confidence %
    """
    try:
        model = joblib.load('crop_recommendation_model.pkl')
        minmax = joblib.load('minmax_scaler.pkl')
        std_scaler = joblib.load('standard_scaler.pkl')
        crop_labels = joblib.load('crop_labels.pkl')
    except FileNotFoundError:
        model, minmax, std_scaler, crop_labels = train_crop_recommendation_model()

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features_minmax = minmax.transform(features)
    features_scaled = std_scaler.transform(features_minmax)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    max_prob = max(probabilities) * 100
    crop_name = crop_labels[prediction]

    return crop_name, max_prob


if __name__ == "__main__":
    train_crop_recommendation_model()

    test_cases = [
        {
            'N': 90, 'P': 42, 'K': 43, 'temperature': 20.87,
            'humidity': 82.00, 'ph': 6.5, 'rainfall': 202.94
        },
        {
            'N': 20, 'P': 30, 'K': 10, 'temperature': 26.5,
            'humidity': 52.5, 'ph': 7.0, 'rainfall': 150.25
        }
    ]

    print("\nTesting prediction with sample data:")
    for i, test in enumerate(test_cases):
        crop_name, probability = predict_crop(
            test['N'], test['P'], test['K'], test['temperature'],
            test['humidity'], test['ph'], test['rainfall']
        )
        print(f"\nTest Case {i + 1}:")
        print(f"Input: {test}")
        print(f"Predicted Crop: {crop_name}")
        print(f"Confidence: {probability:.2f}%")
