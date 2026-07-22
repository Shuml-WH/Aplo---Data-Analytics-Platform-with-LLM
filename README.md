# Aplo Data Analytics Platform

A web-based data analytics platform for non-technical users. Upload CSV data, clean it, build interactive charts, train ML models, and chat with an AI assistant — all through a browser interface.

## Clone

```bash
git clone https://github.com/Shuml-WH/Aplo---Data-Analytics-Platform-with-LLM.git
cd Aplo---Data-Analytics-Platform-with-LLM/Script/Development_code
```

## Installation

**Option A — Docker (Recommended)**

```bash
cd Aplo---Data-Analytics-Platform-with-LLM/Script/Development_code
docker compose up -d
```

Open `http://localhost:3000`. Requires Docker Desktop.

**Option B — Local**

```bash
cd Aplo---Data-Analytics-Platform-with-LLM/Script/Development_code
pip install -r requirements.txt
cd aplo-dashboard
npm install
```

**Run (single command):**
```bash
start_local.bat
```

This launches both backend (`http://localhost:5000`) and frontend (`http://localhost:5173`) in separate windows. Close them to stop.

Requires Python 3.10+, Node 18+, and Ollama with `llama3.2` model.

## Files

| Path | Description |
|------|-------------|
| `app.py` | Flask backend — upload, cleaning, charts, ML, chat |
| `chart_service.py` | Chart building (bar, line, scatter, pie, gauge) |
| `ml_service.py` | ML model training (classification, regression) |
| `timeseries_service.py` | Time-series forecasting (ARIMAX, XGBoost) |
| `ollama_helper.py` | AI chat assistant — prompts, chart parsing |
| `resource_monitor.py` | Background CPU/memory monitoring |
| `requirements.txt` | Python dependencies (pinned) |
| `aplo-dashboard/` | React frontend (Vite + Plotly) |
| `docker-compose.yml` | Docker orchestration |
| `Dockerfile.backend` | Python 3.13 Docker image |
| `Dockerfile.frontend` | Node 22 → Nginx Docker image |
| `nginx.conf` | Nginx reverse proxy config |
| `USER_MANUAL.md` | Full installation and usage guide |

## Usage

Refer to **`USER_MANUAL.md`** or **`USER_MANUAL.pdf`** for detailed instructions on uploading data, building charts, training ML models, and using the AI chat assistant.
