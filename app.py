import gradio as gr
import pickle
import pandas as pd

try:
    with open("bestmodel.pkl", "rb") as f:
        best_model = pickle.load(f)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        "bestmodel.pkl not found. Run assignment_model.py first to generate the model."
    ) from exc

# Prediction function
def predict_purchase(gender, age, salary):
    input_df = pd.DataFrame([[gender, age, salary]],
                            columns=['Gender', 'Age', 'EstimatedSalary'])

    # Predict
    prediction = best_model.predict(input_df)[0]
    prob = best_model.predict_proba(input_df)[0][1]

    # Map prediction
    result = "✅ Will Purchase" if prediction == 1 else "❌ Will Not Purchase"
    return f"{result}\nProbability of purchase: {prob:.2f}"


# Gradio interface
interface = gr.Interface(
    fn=predict_purchase,
    inputs=[
        gr.Radio(choices=["Male", "Female"], label="Gender"),
        gr.Slider(18, 60, step=1, label="Age"),
        gr.Number(label="Estimated Salary")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Social Network Ads Purchase Predictor",
    description="Enter Gender, Age, and Estimated Salary to predict whether the user will purchase."
)
interface.launch()