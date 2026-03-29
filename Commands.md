# Stop everything
docker compose down

# Stop and remove all data (volumes)
docker compose down -v

# Restart just one service
docker compose restart jaeger

# View logs for specific service
docker compose logs -f jaeger
docker compose logs -f otel-collector

# Check resource usage
docker stats --no-stream

# Re-launch after stopping
docker compose up -d

# Show volume details
docker volume inspect fsbs-platform_fsbs-checkpoint-data