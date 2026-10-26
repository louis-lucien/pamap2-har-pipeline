"""
API REST d'inférence temps réel (FastAPI).

POST /predict : reçoit une fenêtre BRUTE de capteurs (256 échantillons
par canal, soit 2,56 s à 100 Hz) et renvoie l'activité reconnue.
C'est exactement le format qu'enverrait un objet connecté / smartphone.

Lancer :  uvicorn api.main:app --reload
Docs interactives :  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict_window

app = FastAPI(
    title="PAMAP2 HAR API",
    description="Reconnaissance d'activité physique en temps réel à partir "
                "de fenêtres brutes d'IMU (main, poitrine, cheville) + cardio.",
    version="1.0.0",
)


class SensorWindow(BaseModel):
    """Fenêtre de signal brut. Chaque canal = liste de 256 échantillons
    (2,56 s à 100 Hz). Canaux requis : {hand,chest,ankle}_acc16_{x,y,z}
    et {hand,chest,ankle}_gyro_{x,y,z}."""
    channels: dict[str, list[float]] = Field(
        ..., description="18 canaux IMU : nom -> 256 échantillons")
    heart_rate: list[float | None] | None = Field(
        None, description="Cardio optionnel (None entre les mesures ~9 Hz)")


class Prediction(BaseModel):
    activity_id: int
    activity_name: str
    confidence: float
    top3: list[dict]
    champion_model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(window: SensorWindow):
    """Renvoie l'activité reconnue et la confiance du modèle."""
    try:
        return predict_window(window.channels, window.heart_rate)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
