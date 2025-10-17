# File: src/train_model.py
# Training script for Stress Predictor with 5 inputs (optimized)

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define the improved model architecture
class StressPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)  # 5 inputs: HRV, Activity, Sleep, BP, RR
        self.fc2 = nn.Linear(16, 8)  # Additional hidden layer
        self.fc3 = nn.Linear(8, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Generate synthetic data for 5 features with realistic stress mapping
np.random.seed(42)
num_samples = 1000
hrv = np.random.normal(60, 20, num_samples)  # HRV (ms)
activity = np.random.normal(10000, 3000, num_samples)  # Steps
sleep = np.random.normal(7, 1.5, num_samples)  # Hours
bp = np.random.normal(120, 15, num_samples)  # Blood Pressure (mmHg)
rr = np.random.normal(16, 2, num_samples)  # Respiration Rate (breaths/min)

# Assign stress levels based on realistic thresholds
stress = np.zeros(num_samples, dtype=int)
stress[(hrv < 50) | (bp > 130) | (rr > 18) | (sleep < 6)] = 2  # High stress
stress[(hrv >= 50) & (hrv < 70) & (bp >= 110) & (bp <= 130) & (rr >= 14) & (rr <= 18) & (sleep >= 6) & (sleep < 8)] = 1  # Medium stress
stress[(hrv >= 70) & (bp < 110) & (rr < 14) & (sleep >= 8)] = 0  # Low stress
# Fill remaining with random to balance
mask = (stress == 0)
stress[mask] = np.random.choice([0, 1, 2], size=mask.sum(), p=[0.4, 0.3, 0.3])

# Create dataset
data = pd.DataFrame({
    'HRV': hrv,
    'Activity': activity,
    'Sleep': sleep,
    'BP': bp,
    'RR': rr,
    'Stress': stress
})

# Prepare features and target
X = data[['HRV', 'Activity', 'Sleep', 'BP', 'RR']].values
y = data['Stress'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Initialize model, loss, and optimizer
model = StressPredictor()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Training loop with early stopping
num_epochs = 300
best_accuracy = 0.0
best_model = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            outputs_test = model(X_test_tensor)
            _, predicted = torch.max(outputs_test.data, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.4f}')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()

# Load best model and evaluate
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Final Test Accuracy: {accuracy:.4f}')

# Save the model and scaler
torch.save(model.state_dict(), 'models/stress_model.pth')
joblib.dump(scaler, 'models/scaler.joblib')
print("Model and scaler saved successfully!")