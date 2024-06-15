import gradio as gr
import pickle
from churn_predictor import ChurnPredictor
import pandas as pd

# Load the model
with open("models/knn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


def model_prediction(
    tenure,
    tenure_category,
    segment1,
    segment2,
    status,
    loyalty_points,
    data_usage_tier,
):
    input_data = {
        "Customer Tenure": [tenure],
        "Tenure Category": [tenure_category],
        "Status": [status],
        "Segment1": [segment1],
        "Segment2": [segment2],
        "Loyalty Points": [loyalty_points],
        "Data Usage Tier": [data_usage_tier],
    }

    input_data = pd.DataFrame(input_data)

    churn_predictor = ChurnPredictor(model)

    prediction = churn_predictor.predict(input_data)
    return prediction  # Assuming a single prediction


iface = gr.Interface(
    fn=model_prediction,
    inputs=[
        gr.Number(label="Customer Tenure"),
        gr.Dropdown(
            choices=["Short-term", "Medium-term", "Long-term"], label="Tenure Category"
        ),
        gr.Radio(choices=["Prepaid", "Postpaid"], label="Segment1"),
        gr.Radio(choices=["Residential", "Corporate", "PRO"], label="Segment2"),
        gr.Radio(
            choices=["Hard Suspended", "Soft Suspended", "Deactive", "Active"],
            label="Status",
        ),
        gr.Number(label="Loyalty Points"),
        gr.Slider(minimum=1, maximum=3, step=1, label="Data Usage Tier"),
    ],
    outputs="text",
)

iface.launch(share=True)
