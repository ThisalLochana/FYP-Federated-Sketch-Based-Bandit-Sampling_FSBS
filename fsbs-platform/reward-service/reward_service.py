"""
FSBS Reward Service — Tier 4: Outcome-Coupled Learning.

This service closes the feedback loop that makes FSBS learn:

  1. Polls Jaeger API for recently collected traces
  2. Analyzes each trace:
     - Error traces → high reward (1.0)
     - Slow traces  → moderate-high reward (0.5-0.8)
     - Complex multi-service traces → moderate reward (0.4)
     - Fast routine traces → low reward (0.1)
  3. Computes arm_index and context vector (same math as sidecar)
  4. POSTs reward to the sidecar HTTP API
  5. Sidecar updates LinUCB + Thompson models

Over time, the bandit LEARNS:
  - "Traces from arm 5 (checkout+payment, high latency) are valuable"
    → increase sampling rate for that arm
  - "Traces from arm 12 (frontend, fast, routine) are not valuable"
    → decrease sampling rate for that arm

This is the core innovation: the sampler adapts to what's actually
important in YOUR system, not just following static rules.
"""

import os
import sys
import time
import logging
import hashlib
import requests
from typing import Optional, Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [reward-service] %(levelname)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('reward-service')


# ── Configuration ──────────────────────────────────────────────

JAEGER_URL = os.environ.get('JAEGER_URL', 'http://jaeger:16686')
SIDECAR_URL = os.environ.get('SIDECAR_URL', 'http://fsbs-sidecar:8081')
POLL_INTERVAL = int(os.environ.get('POLL_INTERVAL', '30'))
TRACES_PER_POLL = int(os.environ.get('TRACES_PER_POLL', '30'))
LOOKBACK_SECONDS = int(os.environ.get('LOOKBACK_SECONDS', '60'))

# Service cluster IDs — must match sidecar's feature_extractor.py
SERVICE_CLUSTER_IDS = {
    "frontend": 0,
    "productcatalogservice": 1,
    "cartservice": 2,
    "currencyservice": 3,
    "checkoutservice": 4,
    "paymentservice": 5,
    "shippingservice": 6,
    "emailservice": 7,
    "recommendationservice": 8,
    "adservice": 9,
    "loadgenerator": 10,
}

LATENCY_BUCKETS_US = [1_000, 5_000, 20_000, 50_000, 200_000, 500_000, 2_000_000]


# ── Feature computation (mirrors sidecar logic) ───────────────

def compute_latency_bucket(duration_us: int) -> int:
    """Same logic as sidecar's FeatureExtractor.compute_latency_bucket."""
    for i, boundary in enumerate(LATENCY_BUCKETS_US):
        if duration_us < boundary:
            return i
    return 7


def compute_topo_hash(service_names: list) -> int:
    """Compute 16-bit topology hash from ordered service list."""
    if not service_names:
        return 0
    path_str = "|".join(
        str(SERVICE_CLUSTER_IDS.get(s, 255)) for s in sorted(service_names)
    )
    h = hashlib.md5(path_str.encode(), usedforsecurity=False).digest()
    return (h[0] << 8) | h[1]


def compute_arm_index(svc_cluster_id: int, topo_hash: int) -> int:
    """Same formula as sidecar: (svc_id << 4) | (topo >> 12) & 0xFF."""
    return ((svc_cluster_id << 4) | (topo_hash >> 12)) & 0xFF


# ── Jaeger trace fetching ─────────────────────────────────────

def fetch_traces(service: str = 'frontend') -> List[Dict]:
    """
    Fetch recent traces from Jaeger REST API.

    Jaeger API: GET /api/traces?service=X&limit=N&lookback=Xs
    Returns list of trace dicts.
    """
    lookback_str = f"{LOOKBACK_SECONDS}s"

    # Try different lookback parameter formats for compatibility
    params = {
        'service': service,
        'limit': TRACES_PER_POLL,
        'lookback': lookback_str,
    }

    try:
        url = f"{JAEGER_URL}/api/traces"
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        traces = data.get('data', [])
        return traces
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to Jaeger at {JAEGER_URL}")
        return []
    except Exception as e:
        logger.error(f"Jaeger fetch error: {e}")
        return []


# ── Trace analysis & reward computation ───────────────────────

def analyze_trace(trace: Dict) -> Optional[Dict[str, Any]]:
    """
    Analyze a Jaeger trace and compute a reward signal.

    Reward rules:
      error trace             → 1.0  (most valuable for debugging)
      very slow (>500ms)      → 0.8  (performance issues)
      slow (>200ms)           → 0.5  (worth investigating)
      complex (≥4 services)   → 0.4  (complex interactions)
      moderate (>50ms)        → 0.2  (some value)
      fast routine            → 0.1  (low debugging value)

    Returns dict with arm_index, context, reward, and metadata.
    Returns None if trace is unparseable.
    """
    spans = trace.get('spans', [])
    processes = trace.get('processes', {})

    if not spans:
        return None

    # ── Find root span (longest duration) ──
    root_span = max(spans, key=lambda s: s.get('duration', 0))
    total_duration_us = root_span.get('duration', 0)

    # ── Check for errors across all spans ──
    has_error = False
    for span in spans:
        for tag in span.get('tags', []):
            key = tag.get('key', '')
            value = tag.get('value')
            if key == 'error' and value is True:
                has_error = True
            if key == 'otel.status_code' and value == 'ERROR':
                has_error = True
            if key == 'http.status_code' and isinstance(value, int) and value >= 500:
                has_error = True

    # ── Identify services involved ──
    services = set()
    for span in spans:
        pid = span.get('processID', '')
        proc = processes.get(pid, {})
        svc = proc.get('serviceName', 'unknown')
        if svc != 'unknown':
            services.add(svc)

    if not services:
        return None

    # ── Compute reward ──
    if has_error:
        reward = 1.0
        reason = "error"
    elif total_duration_us > 500_000:
        reward = 0.8
        reason = "very_slow"
    elif total_duration_us > 200_000:
        reward = 0.5
        reason = "slow"
    elif len(services) >= 4:
        reward = 0.4
        reason = "complex"
    elif total_duration_us > 50_000:
        reward = 0.2
        reason = "moderate"
    else:
        reward = 0.1
        reason = "routine"

    # ── Compute arm index (must match sidecar's computation) ──
    # The sidecar always has parent_services=[] → topo_hash=0
    # We must use the SAME arm index or rewards go to wrong arms
    root_pid = root_span.get('processID', '')
    root_proc = processes.get(root_pid, {})
    root_svc = root_proc.get('serviceName', 'unknown')
    svc_id = SERVICE_CLUSTER_IDS.get(root_svc, 255)

    # Match sidecar: topo_hash=0 since sidecar has no parent info
    arm_index = (svc_id << 4) & 0xFF

    # ── Compute context vector (same as sidecar) ──
    latency_bucket = compute_latency_bucket(total_duration_us)
    context = [
        latency_bucket / 7.0,
        1.0 if has_error else 0.0,
        svc_id / 255.0,
        0.5,  # novelty not available from Jaeger, use neutral value
    ]

    # Build per-service arm entries so ALL services in the trace
    # receive the reward — not just the root service
    arm_entries = []
    for svc in services:
        svc_id = SERVICE_CLUSTER_IDS.get(svc, 255)
        svc_arm = (svc_id << 4) & 0xFF
        svc_context = [
            latency_bucket / 7.0,
            1.0 if has_error else 0.0,
            svc_id / 255.0,
            0.5,
        ]
        arm_entries.append({
            'arm_index': svc_arm,
            'context': svc_context,
        })

    return {
        'arm_entries': arm_entries,
        'reward': reward,
        'reason': reason,
        'trace_id': trace.get('traceID', ''),
        'root_service': root_svc,
        'duration_us': total_duration_us,
        'has_error': has_error,
        'num_services': len(services),
        'num_spans': len(spans),
        'services': sorted(services),
    }


# ── Send reward to sidecar ────────────────────────────────────

def send_reward(arm_index: int, context: list, reward: float) -> bool:
    """POST a reward signal to the sidecar HTTP API."""
    try:
        resp = requests.post(
            f"{SIDECAR_URL}/reward",
            json={
                'arm_index': arm_index,
                'context': context,
                'reward': reward,
            },
            timeout=5,
        )
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to sidecar at {SIDECAR_URL}")
        return False
    except Exception as e:
        logger.error(f"Send reward error: {e}")
        return False


# ── Main loop ─────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("FSBS Reward Service starting")
    logger.info(f"  Jaeger:     {JAEGER_URL}")
    logger.info(f"  Sidecar:    {SIDECAR_URL}")
    logger.info(f"  Poll every: {POLL_INTERVAL}s")
    logger.info(f"  Lookback:   {LOOKBACK_SECONDS}s")
    logger.info("=" * 60)

    # Track which traces we've already processed (avoid double-rewarding)
    processed_traces = set()
    MAX_PROCESSED_CACHE = 10000

    # Wait for infrastructure to be ready
    logger.info("Waiting 20s for Jaeger and sidecar to be ready...")
    time.sleep(20)

    cycle = 0
    total_rewards_sent = 0

    while True:
        cycle += 1
        logger.info(f"── Reward cycle {cycle} ──")

        # Fetch traces from Jaeger
        # Try multiple services to get diverse traces
        all_traces = []
        for svc in ['frontend', 'checkoutservice', 'productcatalogservice']:
            traces = fetch_traces(service=svc)
            all_traces.extend(traces)

        # Deduplicate by trace ID
        seen_ids = set()
        unique_traces = []
        for t in all_traces:
            tid = t.get('traceID', '')
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                unique_traces.append(t)

        logger.info(f"Fetched {len(unique_traces)} unique traces from Jaeger")

        # Analyze and send rewards
        rewards_this_cycle = 0
        reward_summary = {'error': 0, 'very_slow': 0, 'slow': 0,
                          'complex': 0, 'moderate': 0, 'routine': 0}

        for trace in unique_traces:
            tid = trace.get('traceID', '')
            if tid in processed_traces:
                continue

            analysis = analyze_trace(trace)
            if analysis is None:
                continue

            # Send reward to ALL service arms in the trace
            all_sent = True
            for entry in analysis['arm_entries']:
                success = send_reward(
                    entry['arm_index'],
                    entry['context'],
                    analysis['reward'],
                )
                if not success:
                    all_sent = False

            if all_sent:
                rewards_this_cycle += 1
                total_rewards_sent += 1
                processed_traces.add(tid)
                reward_summary[analysis['reason']] = (
                    reward_summary.get(analysis['reason'], 0) + 1
                )

                arms_rewarded = [e['arm_index'] for e in analysis['arm_entries']]
                logger.debug(
                    f"  Reward: arms={arms_rewarded} "
                    f"reward={analysis['reward']:.1f} "
                    f"reason={analysis['reason']} "
                    f"svc={analysis['root_service']} "
                    f"dur={analysis['duration_us']/1000:.0f}ms"
                )

        # Evict old processed traces to prevent memory growth
        if len(processed_traces) > MAX_PROCESSED_CACHE:
            processed_traces.clear()
            logger.info("Cleared processed trace cache")

        logger.info(
            f"Cycle {cycle} complete: "
            f"sent {rewards_this_cycle} rewards "
            f"(total: {total_rewards_sent}) | "
            f"breakdown: {reward_summary}"
        )

        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    main()