# Dashboard image: builds the package into a venv, then copies just the
# venv + dashboard code + dataset into a slim runtime stage. Deliberately
# does NOT copy models/ or reports/ -- those are trained artifacts,
# supplied at run time via the docker-compose volumes (see
# docker-compose.yml and the "train" service in Dockerfile.train). The
# dataset IS baked in (like Dockerfile.train) because the dashboard's
# Overview/EDA tab reads it directly for charts, not just the trained
# artifacts -- found by actually running the built image, not just
# reading the code.
FROM python:3.12-slim AS builder
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY dashboard/ dashboard/
COPY data/ data/

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
