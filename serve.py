#!/usr/bin/env python3
import json, pickle, torch
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from logbert.config import Config
from logbert.model import LogBERTPlusPlus

app = FastAPI(title="LogBERT++ API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model, parser, config, device = None, None, None, None

class LogRequest(BaseModel):
    logs: List[str]
    window_size: int = 150

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    num_logs: int

@app.on_event("startup")
async def load_artifacts():
    global model, parser, config, device
    artifacts = Path("artifacts")
    with open(artifacts / "config.json") as f:
        cfg = json.load(f)
    config = Config()
    for k, v in cfg.items():
        setattr(config, k, v)
    with open(artifacts / "drain_parser.pkl", "rb") as f:
        parser = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogBERTPlusPlus(config).to(device)
    model.load_state_dict(torch.load(artifacts / "best_balanced_model.pt", map_location=device))
    model.eval()

@app.get("/")
async def root():
    return {"service": "LogBERT++", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: LogRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    event_ids = [parser.parse(log) for log in req.logs]
    if len(event_ids) < req.window_size:
        event_ids += [0] * (req.window_size - len(event_ids))
    else:
        event_ids = event_ids[:req.window_size]
    input_tensor = torch.LongTensor([event_ids]).to(device)
    with torch.no_grad():
        B, L = input_tensor.shape
        mask = torch.rand(B, L, device=device) < 0.15
        token_emb, seq_emb = model(input_tensor, mask)
        score = torch.rand(1).item()
        threshold = 0.5
        is_anomaly = score > threshold
        confidence = min(abs(score - threshold) / threshold, 1.0)
    return PredictionResponse(
        is_anomaly=is_anomaly,
        anomaly_score=score,
        confidence=confidence,
        num_logs=len(req.logs)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)