# One image for both the API and the UI (docker-compose.yml picks the command).
FROM python:3.11-slim

ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# editable, so the code finds data/ and results/ next to src/ (paths are relative to the repo)
RUN pip install -e ".[app]"

# Synthetic data and the evaluation caches: the demo runs on the synthetic resumes without
# API calls. Checkpoints and the feedback log go to /app/runs (a volume in compose).
COPY data ./data
COPY results ./results

EXPOSE 8000 8501
CMD ["uvicorn", "shortlist.api:app", "--host", "0.0.0.0", "--port", "8000"]
