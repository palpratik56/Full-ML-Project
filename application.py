import gradio as gr
import pandas as pd
import numpy as np
import os
from src.pipeline.predict import CustomData, PredictPipeline

def predict(gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, math_score, writing_score):
    data = CustomData(
        gender=gender,
        race_ethnicity=ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        math_score=math_score,
        writing_score=writing_score
    )

    df_data = data.get_data_as_data_frame()
    print(df_data)

    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(df_data)
    return np.round(results[0], 3)

#importing the data to get the column values
df = pd.read_csv(os.path.join('artifacts', "train.csv"))
# print(df.gender.unique())

# Create the Gradio interface
iface = gr.Interface(fn=predict,inputs=[
        gr.Dropdown(df.gender.unique().tolist(), label='Gender'),
        gr.Dropdown(df.race_ethnicity.unique().tolist(), label='Ethnicity'),
        gr.Dropdown(df.parental_level_of_education.unique().tolist(), label='Parental Level of Education'),
        gr.Dropdown(df.lunch.unique().tolist(), label='Lunch'),
        gr.Dropdown(df.test_preparation_course.unique().tolist(), label='Test Preparation Course'),
        gr.Number(label='Math Score', minimum=0, maximum=100, step=10),
        gr.Number(label='Writing Score', minimum=0, maximum=100, step=15)
        ],
    outputs=gr.Textbox(label='Predicted Average')
)

if __name__ == "__main__":
    iface.launch(server_port=5001, debug=True, share=True)