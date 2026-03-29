r"""
Collect baseline metrics (100% sampling, no FSBS).

Run this WHILE the baseline stack (docker-compose-baseline-test.yaml) is running.
Wait at least 10 minutes after startup before running.

Usage:
  cd D:\IIT\4th Year\FYP\Implementation_1\fsbs-demo\fsbs-platform
  .\sidecar\venv\Scripts\activate
  python validation\collect_baseline.py
"""

import os
import time
import json
import logging
import requests
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [baseline] %(levelname)s: %(message)s',
)
logger = logging.getLogger()

JAEGER_URL = "http://localhost:16686"


def fetch_traces(service, limit=500, lookback='15m'):
    """Fetch traces from Jaeger REST API."""
    try:
        r = requests.get(
            f"{JAEGER_URL}/api/traces",
            params={'service': service, 'limit': limit, 'lookback': lookback},
            timeout=10,
        )
        return r.json().get('data', [])
    except Exception as e:
        logger.error(f"Fetch failed for {service}: {e}")
        return []


def classify_trace(trace):
    """
    Classify a trace by its characteristics.
    Same classification logic used in validate_fsbs.py
    so the comparison is apples-to-apples.
    """
    spans = trace.get('spans', [])
    processes = trace.get('processes', {})
    if not spans:
        return None

    # Find root span (longest duration)
    root_span = max(spans, key=lambda s: s.get('duration', 0))
    duration_us = root_span.get('duration', 0)
    duration_ms = duration_us / 1000

    # Check for errors across all spans
    has_error = False
    for span in spans:
        for tag in span.get('tags', []):
            k, v = tag.get('key', ''), tag.get('value')
            if k == 'error' and v is True:
                has_error = True
            if k == 'otel.status_code' and v == 'ERROR':
                has_error = True
            if k == 'http.status_code' and isinstance(v, int) and v >= 500:
                has_error = True

    # Count services involved
    services = set()
    for span in spans:
        pid = span.get('processID', '')
        svc = processes.get(pid, {}).get('serviceName', '')
        if svc:
            services.add(svc)

    # Classify (same thresholds as validate_fsbs.py)
    if has_error:
        return {
            'class': 'error', 'value': 1.0,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }
    elif duration_ms > 500:
        return {
            'class': 'very_slow', 'value': 0.8,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }
    elif duration_ms > 200:
        return {
            'class': 'slow', 'value': 0.5,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }
    elif len(services) >= 4:
        return {
            'class': 'complex', 'value': 0.4,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }
    elif duration_ms > 50:
        return {
            'class': 'moderate', 'value': 0.2,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }
    else:
        return {
            'class': 'routine', 'value': 0.1,
            'duration_ms': duration_ms,
            'num_spans': len(spans),
            'num_services': len(services),
        }


def main():
    logger.info("=" * 60)
    logger.info("BASELINE DATA COLLECTION (100% sampling, no FSBS)")
    logger.info("=" * 60)

    # Verify Jaeger is accessible
    try:
        r = requests.get(f"{JAEGER_URL}/api/services", timeout=5)
        services_available = r.json().get('data', [])
        logger.info(f"Jaeger accessible. Services: {services_available}")
    except Exception as e:
        logger.error(f"Cannot reach Jaeger at {JAEGER_URL}: {e}")
        logger.error("Make sure the baseline stack is running!")
        return

    # Fetch from all traced services
    all_traces = []
    for svc in ['frontend', 'checkoutservice', 'productcatalogservice',
                'currencyservice', 'paymentservice']:
        traces = fetch_traces(svc, limit=500, lookback='15m')
        logger.info(f"  {svc}: {len(traces)} traces")
        all_traces.extend(traces)

    # Deduplicate by trace ID
    seen = set()
    unique = []
    for t in all_traces:
        tid = t.get('traceID', '')
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(t)

    logger.info(f"\nTotal unique traces: {len(unique)}")

    if len(unique) == 0:
        logger.error("No traces found! Wait longer or check Jaeger UI.")
        return

    # Classify every trace
    counts = defaultdict(int)
    total_value = 0.0
    total_spans = 0
    durations = []

    for t in unique:
        c = classify_trace(t)
        if c:
            counts[c['class']] += 1
            total_value += c['value']
            total_spans += c['num_spans']
            durations.append(c['duration_ms'])

    # Compute statistics
    avg_value = total_value / len(unique) if unique else 0
    avg_spans = total_spans / len(unique) if unique else 0
    avg_duration = sum(durations) / len(durations) if durations else 0

    result = {
        'timestamp': time.time(),
        'mode': 'baseline_100pct_sampling',
        'description': '100% sampling, no FSBS sidecar, loadgenerator with 10 users',
        'collection_window': '15 minutes',
        'total_traces': len(unique),
        'total_spans_in_traces': total_spans,
        'avg_spans_per_trace': round(avg_spans, 1),
        'avg_duration_ms': round(avg_duration, 1),
        'avg_value_per_trace': round(avg_value, 4),
        'total_value': round(total_value, 2),
        'trace_classes': dict(counts),
        'class_percentages': {
            k: round(v / len(unique) * 100, 1)
            for k, v in counts.items()
        } if unique else {},
    }

    # Print report
    print("\n" + "=" * 60)
    print("BASELINE RESULTS (100% sampling, no FSBS)")
    print("=" * 60)
    print(f"  Total traces:           {result['total_traces']}")
    print(f"  Total spans:            {result['total_spans_in_traces']}")
    print(f"  Avg spans/trace:        {result['avg_spans_per_trace']}")
    print(f"  Avg duration:           {result['avg_duration_ms']:.1f} ms")
    print(f"  Avg value/trace:        {result['avg_value_per_trace']:.4f}")
    print(f"  Total value:            {result['total_value']:.2f}")
    print()
    print("  Trace class distribution:")
    for cls in ['error', 'very_slow', 'slow', 'complex', 'moderate', 'routine']:
        cnt = counts.get(cls, 0)
        pct = cnt / len(unique) * 100 if unique else 0
        print(f"    {cls:<12}: {cnt:>5}  ({pct:>5.1f}%)")
    print()
    print("  This data represents what 100% sampling captures.")
    print("  FSBS should capture fewer TOTAL traces but retain")
    print("  a higher PROPORTION of valuable ones (error, slow).")
    print("=" * 60)

    # Save to file next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'baseline_data.json')
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nBaseline data saved to: {save_path}")
    logger.info("Run validate_fsbs.py after collecting FSBS data to compare.")


if __name__ == '__main__':
    main()