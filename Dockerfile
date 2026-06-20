FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.22 \
    /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY params.yaml ./

COPY models/model.joblib \
    ./models/model.joblib

COPY reports/evaluation/registry_info.json \
    ./reports/evaluation/registry_info.json

RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV MODEL_SOURCE=file
ENV MODEL_PATH=models/model.joblib
ENV REGISTRY_INFO_PATH=reports/evaluation/registry_info.json

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=15s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uv", "run", "--no-sync", "uvicorn", "mlops_prototype.api.main:app", "--host", "0.0.0.0", "--port", "8000"]