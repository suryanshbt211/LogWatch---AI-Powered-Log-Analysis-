# LogBERT++ - Production Log Anomaly Detection

A production-ready log anomaly detection system using hierarchical transformers.

## Quick Start

### Backend API
```bash
# Install dependencies
pip install -r requirements.txt

# Generate artifacts (demo)
python train.py

# Start API server
python serve.py
```

### Frontend
```bash
# Open in browser
open frontend/index.html

# Or use HTTP server
cd frontend && python -m http.server 3000
```

### Docker
```bash
docker build -t logbert-api .
docker run -p 8000:8000 logbert-api
```

## Features
- Hierarchical Transformer architecture
- FastAPI REST API
- Beautiful web interface
- Docker containerized
- Production ready

## API Usage
```python
import requests
response = requests.post("http://localhost:8000/predict", json={
    "logs": ["ERROR: Connection failed", "INFO: Request processed"],
    "window_size": 150
})
print(response.json())
```

## License
MIT