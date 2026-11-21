FROM python:3.11-slim

WORKDIR /app

# System deps for building pandas / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
CMD ["python","scripts/run_pipeline.py","--full"]
