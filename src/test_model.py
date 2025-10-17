# File: src/test_model.py
# Stage 4: Continuous Monitoring & Testing

import torch
import numpy as np
import joblib
import os
from app import StressPredictor  # Import model class from app.py

# Load the trained model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'stress_model.pth')
scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')

if not os.path.exists(model_path):
    raise FileNotFoundError("Model file 'stress_model.pth' not found in models folder!")
if not os.path.exists(scaler_path):
    raise FileNotFoundError("Scaler file 'scaler.joblib' not found in models folder!")

model = StressPredictor()
model.load_state_dict(torch.load(model_path))
model.eval()
scaler = joblib.load(scaler_path)

# Sample test data (5 features: HRV, Activity, Sleep, BP, RR)
test_data = np.array([
    [60.0, 10000, 7.0, 120.0, 16.0],  # Expected: Medium (1)
    [30.0, 5000, 4.0, 150.0, 20.0],   # Expected: High (2)
    [80.0, 15000, 9.0, 100.0, 14.0]   # Expected: Low (0)
])

# Scale and predict
test_scaled = scaler.transform(test_data)
test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

with torch.no_grad():
    predictions = model(test_tensor).argmax(dim=1).numpy()
    expected = np.array([1, 2, 0])  # Expected labels

# Calculate accuracy
accuracy = np.mean(predictions == expected)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print(f"Predictions: {predictions}")
print(f"Expected: {expected}")

# Manual UI Test Instructions
print("\nManual UI Test Instructions:")
print("- Run 'streamlit run app.py' in the src directory.")
print("- Test the following in the app:")
print("  1. Set sliders (e.g., HRV=60, Activity=10000, Sleep=7, BP=120, RR=16) and click 'Predict'.")
print("  2. Check if the stress level (Low/Medium/High) and advice display correctly.")
print("  3. In the chat, type 'I’m stressed' and verify the bot responds empathetically.")