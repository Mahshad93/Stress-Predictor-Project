# File: docs/requirements.py
# Stage 1: Requirements Analysis

# Define requirements as a dictionary
requirements = {
    "societal_need": "Predicting stress for mental health monitoring using wearable data, reducing anxiety and depression",
    "stakeholders": ["General individuals (e.g., students)", "Doctors", "Technology companies"],
    "functional_requirements": {
        "inputs": ["HRV (50-100 ms)", "Activity (0-20000 steps)", "Sleep (4-10 hours)"],
        "outputs": ["Stress level (low/medium/high)", "Simple advice (optional)"]
    },
    "non_functional_requirements": {
        "privacy": "Local data processing",
        "usability": "Simple UI with Streamlit",
        "accuracy": "Minimum 70% ML accuracy",
        "quality": ["Safety (low false alarms)", "Fairness (across age groups)"]
    },
    "constraints": ["No real hardware (simulation)", "3-day timeline"]
}

# Print for verification
print("Project requirements:")
for key, value in requirements.items():
    print(f"{key}: {value}")

# Save to file for GitHub
with open("requirements.txt", "w", encoding="utf-8") as file:
    for key, value in requirements.items():
        file.write(f"{key}: {value}\n")