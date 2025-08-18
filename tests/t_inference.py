from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ==== Carico modello e scaler salvati ====
model = joblib.load("tests/files/occupancy_model.pkl")
scaler = joblib.load("tests/files/occupancy_scaler.pkl")

# ==== Inizializzo FastAPI ====
app = FastAPI()

# ==== Definisco la struttura dei dati di input ====
class OccupancyData(BaseModel):
    S1_Temp: float
    S2_Temp: float
    S3_Temp: float
    S4_Temp: float
    S1_Light: float
    S2_Light: float
    S3_Light: float
    S4_Light: float
    S1_Sound: float
    S2_Sound: float
    S3_Sound: float
    S4_Sound: float
    S5_CO2: float
    S5_CO2_Slope: float
    S6_PIR: int
    S7_PIR: int

# ==== Endpoint di test ====
@app.get("/")
def home():
    return {"message": "Occupancy Prediction API attiva 🚀"}

# ==== Endpoint di predizione ====
@app.post("/predict")
def predict(data: OccupancyData):
    # Trasformo i dati in lista nel giusto ordine
    values = [
        data.S1_Temp, data.S2_Temp, data.S3_Temp, data.S4_Temp,
        data.S1_Light, data.S2_Light, data.S3_Light, data.S4_Light,
        data.S1_Sound, data.S2_Sound, data.S3_Sound, data.S4_Sound,
        data.S5_CO2, data.S5_CO2_Slope, data.S6_PIR, data.S7_PIR
    ]
    
    # Scalo i dati
    values_scaled = scaler.transform([values])
    
    # Faccio la predizione
    prediction = model.predict(values_scaled)[0]
    
    return {"prediction": int(prediction)}