# Stage 2: Design

## Overall Architecture
- **Input Module:** Simulated numerical data (HRV, Activity, Sleep).
- **ML Model:** Simple neural network with PyTorch for stress level classification (low/medium/high).
- **Advice Generator (Optional):** Simple function to generate personalized advice.
- **Output Display:** Results shown with Streamlit and a matplotlib graph.

## Data Flow
- User inputs data via Streamlit sliders (HRV, Activity, Sleep).
- Data is preprocessed (normalized with StandardScaler).
- ML model predicts stress level.
- Optional advice is generated if enabled.
- Results and graph are displayed in the app.

## Technical Choices
- **Data:** Synthetic data generated with numpy (no real hardware).
- **Model:** 2-layer feedforward neural network with torch.
- **UI:** Streamlit for interactive demo.
- **Tools:** numpy, torch, sklearn, matplotlib, streamlit.

## Risks and Mitigations
- **Risk:** Low model accuracy.
  - **Mitigation:** Test with synthetic data and adjust hyperparameters.
- **Risk:** UI crashes.
  - **Mitigation:** Keep Streamlit code simple and test locally.