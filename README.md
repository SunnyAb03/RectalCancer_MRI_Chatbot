# Rectal Cancer MRI Companion

An AI-assisted prototype for explaining structured findings from rectal-cancer MRI reports in plain language. The system combines a React interface, a FastAPI backend, PDF report extraction, structured clinical fields, and a safety-focused chat assistant.

> Research/demo prototype only. It is not a medical device and must not be used to diagnose a patient or make treatment decisions. Users should discuss their report with a qualified clinician.

## What it demonstrates

- Upload a PDF MRI report and extract key fields such as T stage, N stage, CRM, EMVI, mrTRG, and tumour deposits.
- Generate a short, patient-friendly summary.
- Ask follow-up questions about the uploaded report.
- Route chat requests to Gemini, OpenAI, or ZhipuAI when the corresponding API key is configured.
- Store report metadata and chat history locally with SQLite.
- Optionally support literature retrieval through the RAG modules in `backend/rag/`.

## Project structure

```text
backend/          FastAPI API, agents, database models, and RAG modules
frontend/         React + TypeScript web interface
docker-compose.yml Local two-container demo setup
.env.example      Configuration template; never put real keys in Git
```

## Quick start

### 1. Configure API keys

Copy `.env.example` to `.env` and fill in at least `OPENAI_API_KEY` for PDF extraction and summarisation. Add `GOOGLE_API_KEY` and/or `GLM_API_KEY` if you want those chat providers available.

The real `.env` file is intentionally ignored by Git. Never commit API keys, patient reports, or local databases.

### 2. Run the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs` and a health check at `http://localhost:8000/health`.

### 3. Run the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite. The frontend uses `VITE_API_BASE_URL` when it is set and otherwise defaults to `http://localhost:8000`.

### Docker option

```powershell
docker compose up --build
```

The frontend is served on `http://localhost:5173` and the backend on `http://localhost:8000`.

## Privacy and safety

This public repository contains source code and configuration examples only. Runtime databases, uploaded reports, vector indexes, build output, dependency folders, and environment files are ignored. Use anonymised or synthetic reports when demonstrating the prototype.

The assistant is deliberately instructed not to provide a diagnosis, treatment recommendation, prognosis, survival estimate, or recurrence prediction. Outputs are for explanation and research demonstration, not clinical decision-making.

## Known limitations

- PDF extraction quality depends on report layout and image resolution.
- API providers, model names, and pricing can change; configure them for your own environment.
- Authentication and multi-user access control are not implemented in this prototype.
- The default development CORS policy is permissive and should be restricted before any real deployment.

## License

No open-source license has been selected yet. All rights remain with the project owner unless a license is added.
