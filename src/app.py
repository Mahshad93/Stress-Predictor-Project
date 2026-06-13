# File: src/app.py
# Stage 3: Prototype Development - Streamlit App with Gemini API

import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import joblib
import os
import base64
import google.generativeai as genai

st.set_page_config(
    page_title="Stress Level Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    f"Recommendation: Maintain healthy habits. HRV: {int(hrv)} ms, activity: {int(activity)} steps, sleep: {sleep:.1f} hours, BP: {int(bp)} mmHg, RR: {int(rr)} breaths/min.",
    f"Recommendation: Take a short break, practice deep breathing, and prioritize sleep. HRV: {int(hrv)} ms, activity: {int(activity)} steps, sleep: {sleep:.1f} hours, BP: {int(bp)} mmHg, RR: {int(rr)} breaths/min.",
    f"Recommendation: Reduce workload, rest, and consider seeking professional advice if stress symptoms continue. HRV: {int(hrv)} ms, activity: {int(activity)} steps, sleep: {sleep:.1f} hours, BP: {int(bp)} mmHg, RR: {int(rr)} breaths/min."
]
    return f"{warnings[level]}<br>{recommendations[level]}"

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


# Custom CSS for a cleaner portfolio-ready layout
st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background-image: {'linear-gradient(rgba(255,255,255,0.88), rgba(255,255,255,0.88)), url("data:image/jpg;base64,%s")' % base64_image if base64_image else 'none'};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: "Segoe UI", sans-serif;
    }}

    .block-container {{
        max-width: 950px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .main-header {{
        background: linear-gradient(135deg, #1F2937, #4F8EF7);
        color: white;
        padding: 2rem 1.5rem;
        text-align: center;
        border-radius: 22px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }}

    .main-header h1 {{
        font-size: 2.6rem;
        margin-bottom: 0.4rem;
        font-weight: 750;
    }}

    .main-header p {{
        font-size: 1rem;
        color: #E5E7EB;
        margin: 0;
    }}

    .section-card {{
        background-color: rgba(255,255,255,0.94);
        padding: 1.6rem;
        border-radius: 20px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        margin-bottom: 1.4rem;
        border: 1px solid rgba(229,231,235,0.9);
    }}

    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.8rem;
    }}

    .stSlider label {{
        color: #1F2937 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }}

    div[data-testid="stSlider"] {{
        padding-bottom: 0.6rem;
    }}

    .stButton > button {{
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        border: none;
        background-color: #4F8EF7;
        color: white;
        transition: 0.2s;
    }}

    .stButton > button:hover {{
        background-color: #2563EB;
        color: white;
    }}

    .result-box {{
        background-color: white;
        padding: 1.3rem;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        margin-top: 1rem;
        border-left: 7px solid #4F8EF7;
    }}

    .result-title {{
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }}

    .low-result {{
        color: #047857;
        border-left-color: #10B981;
    }}

    .medium-result {{
        color: #B45309;
        border-left-color: #F59E0B;
    }}

    .high-result {{
        color: #B91C1C;
        border-left-color: #EF4444;
    }}

    .recommendation-text {{
        color: #374151;
        font-size: 0.98rem;
        line-height: 1.6;
    }}

    .chat-box {{
        background-color: rgba(255,255,255,0.94);
        padding: 1.2rem;
        border-radius: 18px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #F3F6FA;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #1F2937;
    }}

    .footer-text {{
        text-align: center;
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown(
    """
    <div class="main-header">
        <h1>Stress Level Prediction App</h1>
        <p>Predict stress level using wearable-style health indicators and receive supportive recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar for model explanation
with st.sidebar:
    st.header("About this App")
    st.write(
        "This proof-of-concept application predicts stress levels using wearable-style indicators such as HRV, activity, sleep, blood pressure, and respiration rate."
    )
    st.markdown("**Model type:** Neural Network")
    st.markdown("**Interface:** Streamlit")
    st.markdown("**Optional chatbot:** Gemini API")
    st.info("This app is not intended for medical diagnosis.")
# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'advice' not in st.session_state:
    st.session_state['advice'] = None

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


st.markdown("### Input Health and Activity Indicators")

hrv_input = st.slider(
    'Heart Rate Variability (HRV, ms)',
    min_value=20,
    max_value=100,
    value=60,
    step=1,
    key="hrv"
)

activity_input = st.slider(
    'Daily Activity Level (steps)',
    min_value=0,
    max_value=20000,
    value=10000,
    step=100,
    key="activity"
)

sleep_input = st.slider(
    'Sleep Duration (hours)',
    min_value=4.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
    format="%.1f",
    key="sleep"
)

bp_input = st.slider(
    'Blood Pressure (mmHg)',
    min_value=90,
    max_value=160,
    value=120,
    step=1,
    key="bp"
)

rr_input = st.slider(
    'Respiration Rate (breaths per minute)',
    min_value=12,
    max_value=25,
    value=16,
    step=1,
    key="rr"
)



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

        # Store prediction and advice in session state
        st.session_state['prediction'] = prediction
        st.session_state['advice'] = advice

with col2:
    if st.button('Reset Inputs', key="reset_button"):
        st.session_state['prediction'] = None
        st.session_state['advice'] = None
        st.session_state['chat_history'] = []
        st.rerun()
    
if st.session_state['prediction'] is not None:
    prediction = st.session_state['prediction']
    advice = st.session_state['advice']

    stress_label = ["Low", "Medium", "High"][prediction]
    result_class = ["low-result", "medium-result", "high-result"][prediction]

    st.markdown(
        f"""
        <div class="result-box {result_class}">
            <div class="result-title">Predicted Stress Level: {stress_label}</div>
            <div class="recommendation-text">{advice}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

#Optional chatbot section
with st.expander("Optional Supportive Chatbot"):
    st.write("Share your thoughts if you would like additional supportive suggestions.")

    if st.session_state['prediction'] is None:
        st.info("Please predict your stress level first to enable the chatbot.")
    else:
        user_chat = st.text_area(
            "Your thoughts",
            placeholder="Write how you feel or what is difficult to follow...",
            key="chat_text"
        )

        if st.button("Send Message", key="send_chat_button"):
            if user_chat.strip():
                st.session_state['chat_history'].append(("user", user_chat))
                chat_reply = chat_with_llm(user_chat, st.session_state['prediction'])
                st.session_state['chat_history'].append(("assistant", chat_reply))
            else:
                st.warning("Please write a message before sending.")

        for role, message in st.session_state['chat_history']:
            if role == "user":
                st.markdown(f"**You:** {message}")
            elif role == "assistant":
                st.markdown(f"**Bot:** {message}")


# Footer
st.markdown(
    "<p class='footer-text'>Developed by Mahshad Memarifard | 2025</p>",
    unsafe_allow_html=True
)