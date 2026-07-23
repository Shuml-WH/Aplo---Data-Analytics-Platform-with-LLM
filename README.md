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

## File Tree

```
Development_code/
├── app.py                          # Flask backend (entry point, all API routes)
├── chart_service.py                # Plotly chart builder
├── ml_service.py                   # scikit-learn ML pipeline
├── timeseries_service.py           # ARIMAX / XGBoost forecasting
├── ollama_helper.py                # LLM chat (LangChain + Ollama / llama3.2)
├── resource_monitor.py             # Background CPU/RAM logging
├── requirements.txt                # Python dependencies
├── package.json                    # Root Node deps (E2E test tools)
├── Dockerfile.backend              # Python 3.13-slim image
├── Dockerfile.frontend             # Node 22 build + Nginx serve
├── docker-compose.yml              # Ollama + backend + frontend
├── nginx.conf                      # Reverse proxy (SSE-optimized)
├── start_local.bat                 # One-click local launcher
├── dashboard-dark.css              # Dark theme (standalone copy)
│
├── sample_data/                    # Example CSV datasets
├── uploads/                        # User-uploaded CSV files
├── screenshots/                    # Feature screenshots (40+)
├── test_results/                   # ML benchmark reports
│
└── aplo-dashboard/                 # React frontend
    ├── package.json                # React 19, Vite 8, Plotly 3.6
    ├── vite.config.js              # Vite config (/api proxy → :5000)
    ├── index.html                  # SPA entry
    │
    └── src/
        ├── main.jsx                # React DOM mount
        ├── App.jsx                 # Root: tab navigation + layout
        │
        ├── assets/                 # Static images / SVGs
        ├── styles/
        │   └── dashboard-dark.css  # Full dark theme (1311 lines)
        │
        └── components/
            ├── DataLoadingView.jsx      # Upload, preview, auto-clean
            ├── ChartBuilderView.jsx     # Interactive chart builder
            ├── MLPipelineView.jsx       # ML training wizard
            ├── MLPredictionView.jsx     # Batch predict, what-if, forecast
            └── ChatSidebar.jsx          # AI chatbot (SSE streaming)
```

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React 19 + Vite 8)          │
│                                                         │
│   App.jsx (tab router)                                  │
│   ├── DataLoadingView    ─── upload, preview, clean     │
│   ├── ChartBuilderView   ─── Plotly chart dashboard     │
│   ├── MLPipelineView     ─── 5-step / 3-step wizard    │
│   ├── MLPredictionView   ─── predict, what-if, forecast │
│   └── ChatSidebar        ─── LLM chatbot (SSE stream)  │
│                                                         │
│   State: useState/useEffect (no Redux, no Router)       │
│   Styling: Custom CSS (dark theme, CSS variables)       │
│   HTTP: fetch() → /api/*                                │
└──────────────────────┬──────────────────────────────────┘
                       │  Vite proxy (dev) / Nginx (prod)
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Backend (Flask 3.1 / Python 3.13)       │
│                                                         │
│   app.py ─── routes ───→ service layer                  │
│   ├── /api/upload-csv       → pandas DataFrame          │
│   ├── /api/auto-clean       → missing value handling    │
│   ├── /api/build-chart      → chart_service.py (Plotly) │
│   ├── /api/ml/*             → ml_service.py (sklearn)   │
│   ├── /api/ml/train-timeseries → timeseries_service.py  │
│   ├── /api/chat, /chat-stream → ollama_helper.py        │
│   └── /api/health, /audit-trail, /reset-dataset         │
│                                                         │
│   State: In-memory dicts (no database)                  │
│   SSE: Real-time progress on train/predict/chat         │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   ┌──────────────┐       ┌──────────────────┐
   │  Ollama      │       │  Resource Monitor │
   │  (llama3.2)  │       │  (CPU/RAM log)    │
   └──────────────┘       └──────────────────┘
```

### Key Design Points

| Aspect | Approach |
|--------|----------|
| **Routing** | Tab-based `useState` in `App.jsx` — no React Router |
| **State** | Lifted to `App.jsx` + prop drilling; no Redux/Context |
| **Backend state** | In-memory Python dicts — no database |
| **ML pipeline** | Two tracks: standard (regression/classification) + time-series (ARIMAX/XGBoost) |
| **Progress** | Server-Sent Events (SSE) for long operations |
| **AI chat** | LangChain + local Ollama (llama3.2), intent detection via keyword matching |
| **Deployment** | Docker Compose (3 services) or `start_local.bat` (dev mode) |
| **Proxy** | Vite dev server proxies `/api` → Flask; production uses Nginx |

## Usage

Refer to **`USER_MANUAL.md`** or **`USER_MANUAL.pdf`** for detailed instructions on uploading data, building charts, training ML models, and using the AI chat assistant.
