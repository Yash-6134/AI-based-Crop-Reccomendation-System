from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from model import predict_crop, train_crop_recommendation_model

app = Flask(__name__)

# Crop descriptions for the UI
CROP_INFO = {
    'rice': {
        'description': '''Best time: Kharif (monsoon); needs warm weather and good water.
                        Soil and water: Grows well in clay/loam soil; keep field level; avoid very deep water.
                        Nutrients: Use balanced NPK; do not overuse urea.
                        Major diseases: Blast, bacterial leaf blight, sheath blight.
                        Simple solutions: Use resistant seed, treat seed, keep proper spacing, drain excess water; spray as per local agri advice if disease crosses threshold.
                        Profit idea: Gross = yield per acre × price (MSP/local mandi). Control disease early to protect yield and profit.''',
        'ideal_conditions': 'N: 80-100, P: 40-60, K: 40-45, Temperature: 20-25°C, Humidity: 80-85%, pH: 6.0-7.0, Rainfall: 200-250mm'
    },
    'maize': {
        'description': 'Maize (corn) is a versatile crop used for food, feed, and biofuel. It requires moderate water and nutrients.',
        'ideal_conditions': 'N: 80-100, P: 50-80, K: 40-60, Temperature: 22-30°C, Humidity: 50-80%, pH: 5.5-7.0, Rainfall: 80-100mm'
    },
    'jute': {
        'description': '''Best time: Pre-monsoon to early monsoon.
                        Soil and water: Moist, well-drained soil; avoid standing water.
                        Nutrients: Basal NPK; keep weeds low.
                        Major diseases: Stem/soft rot in waterlogged fields.
                        Simple solutions: Good drainage, wider spacing where needed, seed treatment, field sanitation.
                        Profit idea: Proper retting in clean water gives better fibre grade and price.''',
        'ideal_conditions': 'N: 50-80, P: 40-60, K: 40-50, Temperature: 25-35°C, Humidity: 70-90%, pH: 6.0-7.5, Rainfall: 150-200mm'
    },
    'cotton': {
        'description': 'Cotton is a major fiber crop that grows best in warm climates. It requires moderate water and fertile soil.',
        'ideal_conditions': 'N: 80-120, P: 40-60, K: 40-60, Temperature: 20-30°C, Humidity: 50-80%, pH: 5.5-8.0, Rainfall: 80-120mm'
    },
    'orange': {
        'description': 'Orange trees require subtropical to tropical conditions and moderate water. They are sensitive to extreme cold.',
        'ideal_conditions': 'N: 30-60, P: 30-50, K: 30-50, Temperature: 15-30°C, Humidity: 40-70%, pH: 5.5-7.0, Rainfall: 80-150mm'
    },
    'apple': {
        'description': 'Apple trees require a period of cold dormancy and do best in temperate climates with distinct seasons.',
        'ideal_conditions': 'N: 20-40, P: 20-40, K: 30-50, Temperature: 15-25°C, Humidity: 40-70%, pH: 5.5-6.5, Rainfall: 80-120mm'
    },

    'watermelon': {
        'description': 'Watermelon thrives in warm weather with plenty of sunshine. It needs well-drained soil and consistent moisture.',
        'ideal_conditions': 'N: 20-40, P: 30-50, K: 30-50, Temperature: 22-32°C, Humidity: 50-80%, pH: 6.0-7.0, Rainfall: 40-70mm'
    },
    'grapes': {
        'description': 'Grape vines grow in various climates but prefer warm, dry conditions with good air circulation.',
        'ideal_conditions': 'N: 20-40, P: 20-40, K: 30-50, Temperature: 15-25°C, Humidity: 60-70%, pH: 6.0-7.0, Rainfall: 60-100mm'
    },
    'mango': {
        'description': 'Mango trees thrive in tropical and subtropical climates. They require a dry period for flowering and fruiting.',
        'ideal_conditions': 'N: 20-40, P: 20-40, K: 30-50, Temperature: 24-30°C, Humidity: 50-80%, pH: 5.5-7.5, Rainfall: 70-150mm'
    },
    'banana': {
        'description': 'Banana plants require consistent warmth and moisture. They grow rapidly in humid tropical conditions.',
        'ideal_conditions': 'N: 80-120, P: 20-40, K: 50-70, Temperature: 20-30°C, Humidity: 70-90%, pH: 5.5-7.0, Rainfall: 120-180mm'
    },
  
    'chickpea': {
        'description': 'Chickpea (garbanzo bean) is a cool-season pulse crop that requires moderate water and growing conditions.',
        'ideal_conditions': 'N: 20-40, P: 40-60, K: 20-40, Temperature: 15-30°C, Humidity: 40-70%, pH: 6.0-8.0, Rainfall: 60-100mm'
    },
    'coffee': {
        'description': 'Coffee plants grow best in tropical highlands with moderate rainfall and shade. They prefer rich, well-drained soil.',
        'ideal_conditions': 'N: 80-120, P: 20-40, K: 20-40, Temperature: 15-28°C, Humidity: 50-70%, pH: 6.0-7.0, Rainfall: 150-200mm'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract values from form
            N = float(request.form['nitrogen'])
            P = float(request.form['phosphorus'])
            K = float(request.form['potassium'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            
            # Make prediction
            crop_name, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            
            # Get crop information
            crop_description = CROP_INFO.get(crop_name, {}).get('description', 'No description available.')
            ideal_conditions = CROP_INFO.get(crop_name, {}).get('ideal_conditions', 'Information not available.')
            
            # Prepare response
            result = {
                'crop': crop_name.title(),
                'confidence': round(confidence, 2),
                'description': crop_description,
                'ideal_conditions': ideal_conditions
            }
            
            return render_template('result.html', result=result, input_data={
                'nitrogen': N,
                'phosphorus': P,
                'potassium': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            })
        
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/data-insights')
def data_insights():
    return render_template('data_insights.html')

# API endpoint for programmatic access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Extract values
        N = float(data.get('nitrogen'))
        P = float(data.get('phosphorus'))
        K = float(data.get('potassium'))
        temperature = float(data.get('temperature'))
        humidity = float(data.get('humidity'))
        ph = float(data.get('ph'))
        rainfall = float(data.get('rainfall'))
        
        # Make prediction
        crop_name, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': {
                'crop': crop_name.title(),
                'confidence': round(confidence, 2)
            },
            'crop_info': CROP_INFO.get(crop_name, {})
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    # Check if model exists, if not train it
    if not os.path.exists('crop_recommendation_model.pkl'):
        print("Model not found. Training new model...")
        train_crop_recommendation_model()
    
    app.run(debug=True) 