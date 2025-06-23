from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
job_encoder = label_encoders["job_domain"]

app = FastAPI()

# Define request format
class QuizInput(BaseModel):
    answers: list[int]  # 5 quiz answers in a list

@app.post("/predict")
def predict_job(quiz: QuizInput):
    if len(quiz.answers) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 answers are required.")
    
    try:
        input_data = np.array(quiz.answers).reshape(1, -1)
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = job_encoder.inverse_transform([prediction_encoded])[0]
        return {"prediction": prediction_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
