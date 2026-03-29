FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fsbs/ ./fsbs/
COPY main.py .

# Default configuration (overridable via docker-compose environment)
ENV FSBS_SERVICE_NAME=unknown
ENV FSBS_LISTEN_PORT=4317
ENV FSBS_FORWARD_ENDPOINT=otel-collector:4317
ENV FSBS_ALPHA=1.0
ENV FSBS_THRESHOLD=0.5
ENV FSBS_CONFIDENCE_THRESHOLD=10
ENV FSBS_METRICS_INTERVAL=10

# Expose OTLP gRPC port
EXPOSE 4317

# Run the sidecar
CMD ["python", "-u", "main.py"]