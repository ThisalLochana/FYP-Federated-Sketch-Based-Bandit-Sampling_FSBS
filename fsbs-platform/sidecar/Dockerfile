FROM python:3.12-slim

WORKDIR /app

# Create checkpoint directory inside the container
RUN mkdir -p /data/checkpoints

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fsbs/ ./fsbs/
COPY main.py .

ENV FSBS_SERVICE_NAME=unknown
ENV FSBS_LISTEN_PORT=4317
ENV FSBS_HTTP_PORT=8081
ENV FSBS_FORWARD_ENDPOINT=otel-collector:4317
ENV FSBS_ALPHA=1.0
ENV FSBS_THRESHOLD=0.2
ENV FSBS_CONFIDENCE_THRESHOLD=10
ENV FSBS_METRICS_INTERVAL=10
ENV FSBS_LOG_LEVEL=INFO
ENV FSBS_CHECKPOINT_DIR=/data/checkpoints
ENV FSBS_CHECKPOINT_INTERVAL=60

EXPOSE 4317
EXPOSE 8081

CMD ["python", "-u", "main.py"]