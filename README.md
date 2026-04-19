# FSBS — Federated Sketch-Based Bandit Sampling

An intelligent distributed trace sampling system that uses contextual bandits
(LinUCB) and probabilistic data structures (Count-Min Sketch) to learn which
traces are valuable and which can be safely dropped — reducing observability
costs while retaining 100% of error and anomaly traces.

## What Problem Does This Solve?

Modern microservice systems generate millions of traces per minute. Storing
all of them is expensive. Random sampling (e.g., "keep 10%") is cheap but
blindly discards error traces, slow traces, and rare edge cases — exactly
the traces you need when debugging production incidents.

**FSBS solves this by learning what matters:**
Random Sampling: keeps 10% of traces → misses ~90% of errors
FSBS Sampling: keeps 10% of traces → retains ~100% of errors
+ learns to prioritize slow, complex, and novel patterns

text


## Architecture Overview
┌─────────────────────────────────────────────────────────────────────┐
│ REQUEST PLANE │
│ │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │frontend │ │checkout │ │ payment │ ... │adservice │ │
│ │:8080 │ │:5050 │ │:50051 │ │:9555 │ │
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│ │OTLP │OTLP │OTLP │OTLP │
│ ▼ ▼ ▼ ▼ │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│ │ sidecar │ │ sidecar │ │ sidecar │ │ sidecar │ │
│ │frontend │ │checkout │ │payment │ ... │adservice│ │
│ │:4317 │ │:4317 │ │:4317 │ │:4317 │ │
│ │HTTP:8081│ │HTTP:8082│ │HTTP:8085│ │HTTP:8090│ │
│ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ │
│ │ │ │ │ │
│ └─────────────┴─────────────┴──────────────────┘ │
│ │ OTLP (sampled spans only) │
│ ▼ │
│ ┌──────────────────┐ │
│ │ OTel Collector │ │
│ │ :4317/:4318 │ │
│ └────────┬─────────┘ │
│ │ OTLP │
│ ▼ │
│ ┌──────────────────┐ │
│ │ Jaeger UI │ │
│ │ :16686 │ │
│ └────────┬─────────┘ │
│ │ │
│ ┌────────▼─────────┐ │
│ │ Reward Service │◄── polls Jaeger API │
│ │ (analyzes │ analyzes traces │
│ │ traces, │ computes rewards │
│ │ POSTs rewards) │ POSTs to sidecars │
│ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

text


### Tier 1 — Microservices (Request Plane)

Google's [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo)
microservices demo with 11 services. The services are unmodified — they emit
traces via their existing OpenTelemetry SDK.

| Service | Language | Port | Sidecar HTTP Port |
|---------|----------|------|-------------------|
| frontend | Go | 8080 | 8081 |
| checkoutservice | Go | 5050 | 8082 |
| productcatalogservice | Go | 3550 | 8083 |
| currencyservice | Node.js | 7000 | 8084 |
| paymentservice | Node.js | 50051 | 8085 |
| shippingservice | Go | 50051 | 8086 |
| emailservice | Python | 8080 | 8087 |
| cartservice | .NET | 7070 | 8088 |
| recommendationservice | Python | 8080 | 8089 |
| adservice | Java | 9555 | 8090 |
| loadgenerator | Python | — | 8091 |

### Tier 2 — FSBS Sidecar (Decision Engine)

A Python gRPC server that intercepts OTLP spans from microservices.
For each span it:

1. **Extracts features** (~3µs) — latency bucket, error flag, service ID,
   topology hash → packed into a 4-byte vector
2. **Queries the Count-Min Sketch** (~2µs) — estimates pattern frequency,
   converts to a novelty score (rare patterns get higher scores)
3. **Checks confidence** — if the arm has enough observations, use LinUCB;
   otherwise fall back to Thompson sampling
4. **LinUCB decision** (~5µs) — computes UCB = exploitation + exploration
   bonus. High UCB → SAMPLE, low UCB → DROP
5. **Thompson fallback** (~1µs) — samples from Beta(α,β) distribution.
   Used during cold start until the arm accumulates enough data
6. **Enqueues record** — pushes a 12-byte record to the MPSC queue for
   background sketch updates

Total hot-path overhead: **~12µs per span** (Python prototype).
Architecture target in Rust/C++: <5µs.

#### Arm Index Layout (256 Arms)
Bits [7..5] — latency_bucket (3 bits) = 8 latency ranges
Bits [4..2] — service_id (3 bits) = 8 services (mod 8)
Bits [1..0] — topology (2 bits) = 4 topology patterns
Total: 8 × 8 × 4 = 256 arms

Example mappings:
Frontend (svc=0), fast (lat=0), topo=0 → Arm 0
Frontend (svc=0), slow (lat=6), topo=0 → Arm 192
Cart (svc=2), fast (lat=0), topo=0 → Arm 8

text


### Tier 3 — Local Checkpoint (Crash Recovery)

Every 60 seconds, a background thread saves the complete sidecar state
(~48KB) to a local file. On crash and restart, the state is restored
immediately — no cold start period, LinUCB resumes where it left off.

**File format (~48KB):**
Header (16 bytes): magic "FSBS" + version + timestamp
CMS (8,192 bytes): 4 rows × 512 cols × 4-byte counters
LinUCB (37,888 bytes): 256 arms × (A matrix + A_inv + b vector + n)
Thompson (4,096 bytes): 256 arms × (alpha + beta)

text


**Why not gossip?** Analysis showed that pods running the same service
learn the same patterns independently (same arm indices, same reward
distributions). Cross-service gossip shares arm data the receiver never
uses. Local persistence provides better crash recovery with zero network
overhead.

### Tier 4 — Reward Plane (Outcome-Coupled Learning)

A separate service that closes the feedback loop:
1. Polls Jaeger API every 15 seconds for recently collected traces
2. Analyzes each trace (error? slow? complex? routine?)
3. Assigns a reward score based on trace value
4. POSTs the reward to the specific sidecar(s) whose services participated
5. Sidecar updates LinUCB (A matrix, b vector) and Thompson (α, β)

**Reward scale:**
Error trace → 1.00 (highest value — must never miss)
Extreme slow (>2s) → 0.95
Very slow (>500ms) → 0.90
Slow (>200ms) → 0.70
Complex (≥4 services)→ 0.60
Moderate (>50ms) → 0.30
Routine (fast) → 0.05 (lowest value — safe to drop)

text


Over time, the bandit learns: "error traces are valuable → sample them
more" and "routine fast traces are low value → sample them less."

## Key Results

| Metric | FSBS | Random Sampling |
|--------|------|-----------------|
| Error capture rate | **100%** | ~50% |
| Volume reduction | **~45-55%** fewer spans | ~45-55% fewer spans |
| Avg value per trace | **higher** | lower |
| Sidecar overhead (P50) | ~1,087 µs | N/A |
| Crash recovery time | <1 second | N/A |
| Memory footprint | ~48KB per sidecar | N/A |

## Prerequisites

- **OS**: Windows 10/11 (tested on Windows with Intel i7-1195G7, 16GB RAM)
- **Docker Desktop**: v4.x with 8GB+ memory allocated
- **Python**: 3.10+
- **Git**: 2.x

## Quick Start

### 1. Clone and Setup

```powershell
mkdir C:\fsbs-demo
cd C:\fsbs-demo
git clone https://github.com/GoogleCloudPlatform/microservices-demo.git

mkdir fsbs-platform
cd fsbs-platform
# Copy all project files into this directory
2. Run Unit Tests
PowerShell

cd sidecar
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/test_components.py -v
# Expected: 52 passed, 0 warnings
3. Start the Full Stack
PowerShell

cd ..  # back to fsbs-platform
docker compose up --build -d
docker compose ps
# Expected: 16 containers running
4. Access Services
Service	URL	Purpose
Online Boutique	http://localhost:8080	The demo e-commerce store
Jaeger UI	http://localhost:16686	View distributed traces
FSBS Frontend Sidecar	http://localhost:8081/health	Sidecar health check
FSBS Metrics	http://localhost:8081/metrics	Full metrics JSON
FSBS Arms	http://localhost:8081/arms	Learned arm statistics
FSBS Decisions	http://localhost:8081/decisions?limit=20	Recent decisions
All 11 sidecar HTTP ports: 8081 (frontend) through 8091 (loadgenerator).

5. Monitor the Dashboard
PowerShell

cd sidecar
.\venv\Scripts\activate
python ..\monitoring\dashboard.py
6. Run Validation
PowerShell

# From fsbs-platform directory with venv activated

# Inject mixed traffic (5 minutes)
python validation\anomaly_injector.py

# Benchmark sidecar overhead
python validation\benchmark_overhead.py

# Generate validation report
python validation\validate_fsbs.py
Collecting Baseline for Comparison
PowerShell

# Stop FSBS stack
docker compose down

# Start baseline stack (no sidecar, 100% sampling)
docker compose -f docker-compose-baseline-test.yaml up -d

# Wait 10 minutes for traffic generation
Start-Sleep 600

# Collect baseline data
python validation\collect_baseline.py

# Stop baseline
docker compose -f docker-compose-baseline-test.yaml down

# Start FSBS stack and collect FSBS data
docker compose up --build -d
Start-Sleep 300
python validation\anomaly_injector.py
Start-Sleep 120
python validation\validate_fsbs.py
# Report will include baseline comparison
Project Structure
text

fsbs-platform/
├── docker-compose.yaml                  ← FSBS stack (16 containers)
├── docker-compose-baseline-test.yaml    ← Baseline stack (no sidecar)
├── README.md                            ← This file
│
├── otel-config/
│   └── otel-collector-config.yaml       ← OTel Collector pipeline config
│
├── sidecar/                             ← FSBS Sidecar (Python)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                          ← gRPC server + HTTP API entry point
│   ├── fsbs/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py         ← Span → 4-byte feature vector
│   │   ├── count_min_sketch.py          ← 8KB probabilistic frequency store
│   │   ├── linucb.py                    ← Contextual bandit (256 arms)
│   │   ├── thompson.py                  ← Beta-distribution cold-start fallback
│   │   ├── mpsc_queue.py                ← Lock-free queue for async updates
│   │   ├── sampler.py                   ← Decision orchestrator
│   │   ├── http_api.py                  ← REST API for metrics and rewards
│   │   └── checkpoint.py                ← Local state persistence (48KB)
│   └── tests/
│       ├── __init__.py
│       ├── test_components.py           ← 52 unit/integration tests
│       └── README.md                    ← Test documentation
│
├── reward-service/                      ← Reward feedback loop
│   ├── Dockerfile
│   ├── requirements.txt
│   └── reward_service.py                ← Jaeger → analyze → reward → sidecar
│
├── monitoring/
│   └── dashboard.py                     ← Live terminal dashboard
│
└── validation/
    ├── anomaly_injector.py              ← Generate mixed traffic with errors
    ├── benchmark_overhead.py            ← Measure sidecar latency per span
    ├── validate_fsbs.py                 ← FSBS vs random comparison report
    ├── collect_baseline.py              ← Collect 100% sampling baseline data
    ├── test_independent_learning.py     ← Verify arm convergence across pods
    ├── baseline_data.json               ← (generated) baseline metrics
    ├── validation_report.txt            ← (generated) final report
    └── validation_data.json             ← (generated) raw validation data
Configuration
All sidecar settings are configurable via environment variables:

Variable	Default	Description
FSBS_SERVICE_NAME	unknown	Name of the attached service
FSBS_LISTEN_PORT	4317	gRPC port for receiving OTLP spans
FSBS_HTTP_PORT	8081	HTTP port for metrics/rewards API
FSBS_FORWARD_ENDPOINT	otel-collector:4317	Where to send sampled spans
FSBS_ALPHA	1.0	LinUCB exploration parameter
FSBS_THRESHOLD	0.5	UCB score threshold for SAMPLE decision
FSBS_CONFIDENCE_THRESHOLD	10	Min observations before trusting LinUCB
FSBS_CHECKPOINT_DIR	(empty)	Directory for checkpoint files
FSBS_CHECKPOINT_INTERVAL	60	Seconds between checkpoint saves
FSBS_METRICS_INTERVAL	10	Seconds between metrics log lines
FSBS_LOG_LEVEL	INFO	Logging level
Key Algorithms
LinUCB (Contextual Bandit)
Each of 256 arms maintains a 4×4 matrix A and 4-element vector b.
The UCB score for context vector x is:

text

θ* = A⁻¹ · b                          (expected reward)
UCB = xᵀθ* + α · √(xᵀ · A⁻¹ · x)    (exploitation + exploration)
If UCB ≥ threshold → SAMPLE, else → DROP.

A⁻¹ is maintained incrementally via Sherman-Morrison rank-1 updates
(O(n²) instead of O(n³) matrix inversion).

Count-Min Sketch (Novelty Detection)
4 rows × 512 columns of 4-byte counters (8KB total).
Query returns minimum counter across 4 hash functions.
Novelty score = 1.0 / (1 + log(estimate + 1)).
Periodic decay (factor=0.9 every 5 min) prevents counter saturation.

Thompson Sampling (Cold Start)
Beta(α, β) distribution per arm. Initially uniform Beta(1,1).
Sample p ~ Beta(α, β), compare to threshold.
Transitions automatically to LinUCB as observations accumulate.

Docker Commands Reference
PowerShell

# Start everything
docker compose up --build -d

# View all container status
docker compose ps

# View sidecar logs
docker compose logs -f fsbs-sidecar-frontend

# View reward service logs
docker compose logs -f reward-service

# Restart just one sidecar (after code changes)
docker compose up --build -d fsbs-sidecar-frontend

# Stop everything
docker compose down

# Stop and remove all data including checkpoint volumes
docker compose down -v

# Simulate a crash and recovery
docker kill fsbs-sidecar-frontend

# Check resource usage
docker stats --no-stream

# Check a specific service's OTLP connection
docker compose logs paymentservice | tail -30
HTTP API Reference
GET /health
JSON

{"status": "ok"}
GET /metrics
JSON

{
  "sampler": {
    "total_spans": 12847,
    "sampled_spans": 6523,
    "sample_rate": 0.508,
    "thompson_decisions": 284,
    "linucb_decisions": 305,
    "forced_samples": 42,
    "rewards_received": 347,
    "active_arms": 18,
    "confident_arms": 12,
    "avg_reward": 0.419,
    "checkpoint": {
      "saves_completed": 28,
      "last_save_size_bytes": 48156,
      "restore_success": true
    }
  },
  "service": {
    "total_spans_in": 12847,
    "total_spans_out": 6523,
    "forward_errors": 0,
    "trace_cache_size": 3207,
    "trace_cache_hits": 42695
  }
}
GET /arms
JSON

{
  "active_arms": 18,
  "confident_arms": 12,
  "arms": [
    {
      "arm_index": 2,
      "n_observations": 16871,
      "confident": true,
      "thompson_mean": 0.419,
      "thompson_alpha": 7063.5,
      "thompson_beta": 9808.5
    }
  ]
}
GET /decisions?limit=10
JSON

{
  "count": 10,
  "decisions": [
    {
      "trace_id": "abc123def456",
      "service": "frontend",
      "operation": "HTTP GET /",
      "duration_us": 45000,
      "arm_index": 2,
      "method": "linucb",
      "score": 0.7234,
      "should_sample": true
    }
  ]
}
POST /reward
JSON

// Request:
{
  "arm_index": 5,
  "context": [0.5, 0.0, 0.1, 0.8],
  "reward": 0.9
}

// Response:
{
  "status": "ok",
  "arm_index": 5,
  "new_n_observations": 15,
  "confident": true
}
Troubleshooting
Problem	Solution
Sidecar shows 0 spans	Check service OTLP endpoint — Python/Java need http:// prefix, Go gRPC does not
Pass rate >70%	Increase FSBS_THRESHOLD from 0.05 to 0.5 in docker-compose.yaml
Learning is slow	Check reward service logs — average reward should be >0.2 after 30 min
No traces in Jaeger	Wait 2-3 minutes for loadgenerator traffic
forward_errors > 0	OTel Collector may be down: docker compose restart otel-collector
Dashboard shows 0 spans	Verify microservices point to fsbs-sidecar:4317 not otel-collector:4317
Frontend slow (>5s)	Normal with 16 containers on 16GB RAM. Increase Docker memory
Checkpoint not saving	Check FSBS_CHECKPOINT_DIR is set and volume is mounted
Tests fail numpy error	Use numpy==1.26.4 instead of numpy==2.1.0 in requirements.txt
Known Issues and Fixes
OTLP Endpoint Configuration by SDK Language
Different language SDKs require different OTLP endpoint formats:

Language	Correct Format	Example
Go (gRPC)	No http:// prefix	fsbs-sidecar-frontend:4317
Python	Requires http:// prefix	http://fsbs-sidecar-emailservice:4317
Java	Requires http:// prefix	http://fsbs-sidecar-adservice:4317
Node.js	Requires http:// prefix	http://fsbs-sidecar-paymentservice:4317
.NET	Requires http:// prefix	http://fsbs-sidecar-cartservice:4317
Cold Start Behavior
On first startup (no checkpoint), the system uses Thompson sampling for
all decisions until each arm accumulates 10+ observations. With 50 users
in the loadgenerator, confident arms typically appear within 5-10 minutes.
LinUCB becomes the primary decision method within 20-30 minutes.

Note on -v flag: docker compose down -v removes checkpoint volumes.
The system cold-starts with Thompson sampling and transitions to LinUCB
within ~10 minutes as the reward service sends feedback to all 11 sidecars.