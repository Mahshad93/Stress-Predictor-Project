# File: src/app.py
# Stage 3: Prototype Development - Streamlit App with Gemini API

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import joblib
import os
import base64
import google.generativeai as genai

# Define the enhanced model architecture with 5 inputs (HRV, Activity, Sleep, BP, RR)
class StressPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)  # 5 inputs: HRV, Activity, Sleep, BP, RR
        self.fc2 = nn.Linear(16, 8)  # Additional hidden layer
        self.fc3 = nn.Linear(8, 3)   # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the trained model and scaler
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'models', 'stress_model.pth')
scaler_path = os.path.join(base_dir, 'models', 'scaler.joblib')

if not os.path.exists(model_path):
    st.error("Model file 'stress_model.pth' not found in models folder!")
    st.stop()
if not os.path.exists(scaler_path):
    st.error("Scaler file 'scaler.joblib' not found in models folder!")
    st.stop()

model = StressPredictor()
model.load_state_dict(torch.load(model_path))
model.eval()
scaler = joblib.load(scaler_path)

# Updated advice function with personalization
def generate_advice(level, hrv, activity, sleep, bp, rr):
    warnings = ["Low stress: Good job, but stay vigilant!", "Medium stress: Attention needed!", "High stress: Urgent action required!"]
    recommendations = [
        f"Recommendation: Maintain habits. With HRV {hrv:.1f}, activity {activity}, sleep {sleep:.1f}, BP {bp:.1f}, RR {rr:.1f}, consider mindfulness."
        , f"Recommendation: Take a break. With HRV {hrv:.1f}, activity {activity}, sleep {sleep:.1f}, BP {bp:.1f}, RR {rr:.1f}, try deep breathing and more sleep."
        , f"Recommendation: Consult a doctor. With HRV {hrv:.1f}, activity {activity}, sleep {sleep:.1f}, BP {bp:.1f}, RR {rr:.1f}, reduce workload and relax."
    ]
    return f"**Warning:** {warnings[level]} {recommendations[level]}"

# Real LLM chat using Gemini API
def chat_with_llm(user_input, stress_level):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "The chatbot is currently disabled because the Gemini API key is not configured."

    try:
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel("gemini-2.0-flash")

        stress_labels = ["low", "medium", "high"]
        stress_text = stress_labels[stress_level] if stress_level in [0, 1, 2] else "unknown"

        prompt = f"""
        You are a supportive chatbot for stress management.
        The user's predicted stress level is {stress_text}.
        Respond empathetically, give short and practical stress-management suggestions, and avoid medical diagnosis.
        User: {user_input}
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Chatbot error: {str(e)}"

# Load background image as base64 with fallback
background_path = os.path.join(base_dir, 'background.jpg')
base64_image = ""
if os.path.exists(background_path):
    try:
        with open(background_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.warning(f"Error loading background image: {e}. Using default background.")
else:
    st.warning("Background image 'background.jpg' not found, using default background.")

# Custom CSS for layout and styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: {'url("data:image/jpg;base64,%s")' % base64_image if base64_image else 'none'};
        background-size: cover;
        background-position: center;
        padding: 20px;
        font-size: 1.8em;
    }}
    .header {{
        background-color: #697682;
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 5px;
    }}
    .stSlider, .stButton {{
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-size: 1.6em;
    }}
    .stButton:hover {{
        background-color: #2980b9;
    }}
    .result-box {{
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }}
    .green-text {{
        color: #27ae60;
        font-weight: bold;
    }}
    .yellow-text {{
        color: #f39c12;
        font-weight: bold;
    }}
    .red-text {{
        color: #c0392b;
        font-weight: bold;
    }}
    .chat-box {{
        background-color: #ecf0f1;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<div class="header"><h1>Stress Level Predictor</h1></div>', unsafe_allow_html=True)

# Sidebar for model explanation
with st.sidebar:
    st.header("About the Model")
    st.write("This is an innovation-driven AI tool that predicts stress levels from wearable data (HRV, Activity, Sleep, BP, RR) using a neural network. Developed as a proof-of-concept for AI Systems Engineering.")

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Inputs as questions with explanations
hrv_input = st.slider('What is your Heart Rate Variability (HRV, ms)?', 20.0, 100.0, 60.0, key="hrv")
activity_input = st.slider('Daily Activity level (steps)?', 0, 20000, 10000, key="activity")
sleep_input = st.slider('Hours of Sleep last night?', 4.0, 10.0, 7.0, key="sleep")
bp_input = st.slider('Blood Pressure (mmHg, e.g., 120/80 average)?', 90.0, 160.0, 120.0, key="bp")  # Average BP
rr_input = st.slider('Respiration Rate (breaths per minute)?', 12.0, 25.0, 16.0, key="rr")  # Normal range

col1, col2 = st.columns(2)
with col1:
    if st.button('Predict', key="predict_button"):
        # Prepare input data with 5 features
        input_data = np.array([[hrv_input, activity_input, sleep_input, bp_input, rr_input]])
        input_scaled = scaler.transform(input_data)
        input_t = torch.tensor(input_scaled, dtype=torch.float32)
        
        # Predict
        with torch.no_grad():
            prediction = model(input_t).argmax().item()
            advice = generate_advice(prediction, hrv_input, activity_input, sleep_input, bp_input, rr_input)
        
        # Store prediction in session state
        st.session_state['prediction'] = prediction
        
        # Display results with color
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        color_class = ["green-text", "yellow-text", "red-text"][prediction]
        st.write(f'<span class="{color_class}">Predicted Stress Level: {["Low", "Medium", "High"][prediction]}</span>', unsafe_allow_html=True)
        st.write(f'<span class="{color_class}">{advice}</span>', unsafe_allow_html=True)
        
        # Stress level progress bar
        st.progress(prediction / 2)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Update and display plot with normalized values
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized_values = [hrv_input / 10, activity_input / 1000, sleep_input, bp_input / 100, rr_input / 10]
        ax.bar(['HRV/10', 'Act/1000', 'Sleep', 'BP/100', 'RR/10'], normalized_values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f1c40f'])
        ax.set_ylabel('Normalized Values')
        st.pyplot(fig, clear_figure=True)

    # Chat section with chat_input and session_state
    st.markdown('<div class="chat-box">', unsafe_allow_html=True)
    st.write("If you can't follow the advice, share your feelings here:")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    user_chat = st.chat_input("Your thoughts...")
    if user_chat and st.session_state['prediction'] is not None:
        st.session_state['chat_history'].append(("user", user_chat))
        chat_reply = chat_with_llm(user_chat, st.session_state['prediction'])
        st.session_state['chat_history'].append(("assistant", chat_reply))

    # Display chat history
    for role, message in st.session_state['chat_history']:
        if role == "user":
            st.write(f"<span style='color: #34495e;'>You: {message}</span>", unsafe_allow_html=True)
        elif role == "assistant":
            st.write(f"<span style='color: #2c3e50;'>Bot: {message}</span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.button('Reset Inputs', key="reset_button"):
        st.session_state['prediction'] = None
        st.session_state['chat_history'] = []
        st.rerun()

# Footer
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Developed by Mahshad Memarifard | 2025</p>", unsafe_allow_html=True)