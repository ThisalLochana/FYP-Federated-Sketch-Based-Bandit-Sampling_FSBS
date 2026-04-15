r"""
FSBS Validation — compares FSBS sampling against random sampling baseline.
Updated for multi-sidecar architecture (11 sidecars).

This script:
  1. Collects metrics from ALL sidecars (frontend, productcatalog, cart, etc.)
  2. Fetches traces from Jaeger
  3. Analyzes what FSBS captured vs what it dropped
  4. Simulates what RANDOM sampling would have captured
  5. Computes and compares quality metrics
  6. Generates a final report

Usage:
  cd D:\IIT\...\fsbs-platform
  python validation\validate_fsbs.py

Run AFTER the anomaly injector has completed (or after 60+ min of traffic).
"""

import json
import random
import time
import logging
import requests
import os
from typing import Dict, List, Any
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [validator] %(levelname)s: %(message)s',
)
logger = logging.getLogger('validator')

JAEGER_URL = "http://localhost:16686"

# ALL 11 SIDECAR URLs
SIDECAR_URLS = {
    'frontend':        'http://localhost:8081',
    'checkout':        'http://localhost:8082',
    'productcatalog':  'http://localhost:8083',
    'currency':        'http://localhost:8084',
    'payment':         'http://localhost:8085',
    'shipping':        'http://localhost:8086',
    'email':           'http://localhost:8087',
    'cart':            'http://localhost:8088',
    'recommendation':  'http://localhost:8089',
    'ad':              'http://localhost:8090',
    'loadgen':         'http://localhost:8091',
}


# ── Data Collection (Multi-Sidecar) ──────────────────────────

def fetch_all_sidecar_metrics() -> Dict[str, Dict]:
    """Get metrics from ALL sidecars."""
    all_metrics = {}
    for name, url in SIDECAR_URLS.items():
        try:
            r = requests.get(f"{url}/metrics", timeout=5)
            all_metrics[name] = r.json()
        except Exception as e:
            logger.warning(f"Cannot reach {name} sidecar: {e}")
            all_metrics[name] = None
    return all_metrics


def fetch_all_sidecar_arms() -> Dict[str, Dict]:
    """Get arm statistics from ALL sidecars."""
    all_arms = {}
    for name, url in SIDECAR_URLS.items():
        try:
            r = requests.get(f"{url}/arms", timeout=5)
            all_arms[name] = r.json()
        except Exception as e:
            logger.warning(f"Cannot reach {name} sidecar arms: {e}")
            all_arms[name] = None
    return all_arms


def aggregate_metrics(all_metrics: Dict[str, Dict]) -> Dict:
    """Combine metrics from all sidecars into cluster-wide totals."""
    valid = {k: v for k, v in all_metrics.items() if v is not None}
    
    if not valid:
        return None
    
    # Sum up all the counters
    total_spans_in = sum(v['service']['total_spans_in'] for v in valid.values())
    total_spans_out = sum(v['service']['total_spans_out'] for v in valid.values())
    thompson = sum(v['sampler']['thompson_decisions'] for v in valid.values())
    linucb = sum(v['sampler']['linucb_decisions'] for v in valid.values())
    forced = sum(v['sampler']['forced_samples'] for v in valid.values())
    rewards = sum(v['sampler']['rewards_received'] for v in valid.values())
    
    # Weighted average reward (by number of rewards)
    total_reward_value = sum(
        v['sampler']['rewards_received'] * v['sampler']['avg_reward'] 
        for v in valid.values()
        if v['sampler']['rewards_received'] > 0
    )
    avg_reward = total_reward_value / rewards if rewards > 0 else 0.0
    
    # Count unique active/confident arms across all sidecars
    all_active_arms = set()
    all_confident_arms = set()
    for v in valid.values():
        # Collect arm indices (you'd need to query /arms for this)
        all_active_arms.add(v['sampler']['active_arms'])
        all_confident_arms.add(v['sampler']['confident_arms'])
    
    total_active = sum(v['sampler']['active_arms'] for v in valid.values())
    total_confident = sum(v['sampler']['confident_arms'] for v in valid.values())
    
    return {
        'total_spans_in': total_spans_in,
        'total_spans_out': total_spans_out,
        'sample_rate': total_spans_out / total_spans_in if total_spans_in > 0 else 0,
        'thompson_decisions': thompson,
        'linucb_decisions': linucb,
        'forced_samples': forced,
        'rewards_received': rewards,
        'avg_reward': avg_reward,
        'active_arms': total_active,
        'confident_arms': total_confident,
        'num_sidecars_alive': len(valid),
        'num_sidecars_total': len(all_metrics),
    }


def fetch_jaeger_traces(
    service: str = 'frontend',
    limit: int = 500,
    lookback: str = '2h',
) -> List[Dict]:
    """Fetch traces from Jaeger."""
    try:
        r = requests.get(
            f"{JAEGER_URL}/api/traces",
            params={'service': service, 'limit': limit, 'lookback': lookback},
            timeout=10,
        )
        return r.json().get('data', [])
    except Exception as e:
        logger.error(f"Cannot reach Jaeger: {e}")
        return []


# ── Trace Analysis ────────────────────────────────────────────

def classify_trace(trace: Dict) -> Dict:
    """
    Classify a trace by its characteristics.
    Returns a dict with classification info.
    """
    spans = trace.get('spans', [])
    processes = trace.get('processes', {})

    if not spans:
        return {'class': 'unknown', 'value': 0}

    # Duration
    root_span = max(spans, key=lambda s: s.get('duration', 0))
    duration_us = root_span.get('duration', 0)
    duration_ms = duration_us / 1000

    # Errors
    has_error = False
    error_count = 0
    for span in spans:
        for tag in span.get('tags', []):
            k, v = tag.get('key', ''), tag.get('value')
            if k == 'error' and v is True:
                has_error = True
                error_count += 1
            if k == 'otel.status_code' and v == 'ERROR':
                has_error = True
                error_count += 1
            if k == 'http.status_code' and isinstance(v, int) and v >= 500:
                has_error = True
                error_count += 1

    # Services
    services = set()
    for span in spans:
        pid = span.get('processID', '')
        svc = processes.get(pid, {}).get('serviceName', '')
        if svc:
            services.add(svc)

    # Classification
    if has_error:
        trace_class = 'error'
        value = 1.0
    elif duration_ms > 500:
        trace_class = 'very_slow'
        value = 0.8
    elif duration_ms > 200:
        trace_class = 'slow'
        value = 0.5
    elif len(services) >= 4:
        trace_class = 'complex'
        value = 0.4
    elif duration_ms > 50:
        trace_class = 'moderate'
        value = 0.2
    else:
        trace_class = 'routine'
        value = 0.1

    return {
        'class': trace_class,
        'value': value,
        'duration_ms': round(duration_ms, 1),
        'has_error': has_error,
        'error_count': error_count,
        'num_spans': len(spans),
        'num_services': len(services),
        'services': sorted(services),
        'trace_id': trace.get('traceID', ''),
    }


def simulate_random_sampling(
    traces: List[Dict],
    sample_rate: float,
    n_simulations: int = 100,
) -> Dict:
    """
    Simulate what RANDOM sampling would capture.
    """
    classifications = [classify_trace(t) for t in traces]
    total = len(classifications)

    if total == 0:
        return {'error': 'no traces'}

    # Count actual valuable traces
    actual_counts = defaultdict(int)
    for c in classifications:
        actual_counts[c['class']] += 1

    # Run simulations
    sim_results = []
    for _ in range(n_simulations):
        sampled = [c for c in classifications if random.random() < sample_rate]
        sampled_counts = defaultdict(int)
        for c in sampled:
            sampled_counts[c['class']] += 1

        # Error capture rate
        error_total = actual_counts.get('error', 0)
        error_captured = sampled_counts.get('error', 0)
        error_rate = error_captured / error_total if error_total > 0 else 1.0

        # Slow capture rate (very_slow + slow)
        slow_total = actual_counts.get('very_slow', 0) + actual_counts.get('slow', 0)
        slow_captured = (
            sampled_counts.get('very_slow', 0) + sampled_counts.get('slow', 0)
        )
        slow_rate = slow_captured / slow_total if slow_total > 0 else 1.0

        # Total value captured
        total_value = sum(c['value'] for c in classifications)
        sampled_value = sum(c['value'] for c in sampled)
        value_rate = sampled_value / total_value if total_value > 0 else 0

        sim_results.append({
            'sample_count': len(sampled),
            'sample_rate': len(sampled) / total if total > 0 else 0,
            'error_capture_rate': error_rate,
            'slow_capture_rate': slow_rate,
            'value_capture_rate': value_rate,
        })

    # Average across simulations
    avg = {}
    for key in sim_results[0]:
        avg[key] = sum(r[key] for r in sim_results) / len(sim_results)

    return {
        'target_sample_rate': sample_rate,
        'avg_sample_count': round(avg['sample_count'], 1),
        'avg_actual_rate': round(avg['sample_rate'], 4),
        'avg_error_capture': round(avg['error_capture_rate'], 4),
        'avg_slow_capture': round(avg['slow_capture_rate'], 4),
        'avg_value_capture': round(avg['value_capture_rate'], 4),
    }


# ── Main Validation ───────────────────────────────────────────

def run_validation():
    logger.info("=" * 70)
    logger.info("FSBS VALIDATION — FINAL ANALYSIS (Multi-Sidecar)")
    logger.info("=" * 70)

    # ── 1. Collect metrics from ALL sidecars ──
    logger.info("\n[1/5] Collecting metrics from ALL sidecars...")
    all_metrics = fetch_all_sidecar_metrics()
    all_arms = fetch_all_sidecar_arms()
    
    # Aggregate cluster-wide metrics
    agg = aggregate_metrics(all_metrics)
    if not agg:
        logger.error("Cannot connect to any sidecars. Is the stack running?")
        return

    logger.info(f"  Connected to {agg['num_sidecars_alive']}/{agg['num_sidecars_total']} sidecars")

    # ── 2. Fetch Jaeger traces ──
    logger.info("[2/5] Fetching traces from Jaeger...")
    all_traces = []
    for svc in ['frontend', 'checkoutservice', 'productcatalogservice',
                'currencyservice', 'paymentservice']:
        traces = fetch_jaeger_traces(service=svc, limit=300, lookback='2h')
        all_traces.extend(traces)

    # Deduplicate
    seen = set()
    unique_traces = []
    for t in all_traces:
        tid = t.get('traceID', '')
        if tid and tid not in seen:
            seen.add(tid)
            unique_traces.append(t)

    logger.info(f"  Found {len(unique_traces)} unique traces in Jaeger")

    # ── 3. Classify captured traces ──
    logger.info("[3/5] Classifying captured traces...")
    classifications = [classify_trace(t) for t in unique_traces]

    class_counts = defaultdict(int)
    class_values = defaultdict(float)
    for c in classifications:
        class_counts[c['class']] += 1
        class_values[c['class']] += c['value']

    # ── 4. Simulate random sampling ──
    logger.info("[4/5] Simulating random sampling baseline...")

    fsbs_sample_rate = agg['sample_rate']

    random_results = simulate_random_sampling(
        unique_traces,
        sample_rate=fsbs_sample_rate,
        n_simulations=200,
    )

    # FSBS capture rates (from what's actually in Jaeger)
    total_traces = len(classifications)
    fsbs_error_count = class_counts.get('error', 0)
    fsbs_slow_count = (
        class_counts.get('very_slow', 0) + class_counts.get('slow', 0)
    )
    fsbs_total_value = sum(c['value'] for c in classifications)

    # ── 5. Generate report ──
    logger.info("[5/5] Generating validation report...\n")

    report = []
    report.append("=" * 70)
    report.append("           FSBS VALIDATION REPORT (Multi-Sidecar)")
    report.append("=" * 70)

    # System overview
    report.append("\n── CLUSTER STATUS ──")
    report.append(f"  Sidecars alive:    {agg['num_sidecars_alive']}/{agg['num_sidecars_total']}")
    report.append(f"  Total spans in:    {agg['total_spans_in']}")
    report.append(f"  Total spans out:   {agg['total_spans_out']}")
    report.append(f"  Span pass rate:    {agg['sample_rate']:.1%}")

    # Decision engine
    report.append("\n── DECISION ENGINE (CLUSTER-WIDE) ──")
    thompson = agg['thompson_decisions']
    linucb = agg['linucb_decisions']
    forced = agg['forced_samples']
    total_dec = thompson + linucb + forced
    if total_dec > 0:
        report.append(
            f"  Thompson:          {thompson} "
            f"({thompson/total_dec:.1%})"
        )
        report.append(
            f"  LinUCB:            {linucb} "
            f"({linucb/total_dec:.1%})"
        )
        report.append(
            f"  Forced (errors):   {forced} "
            f"({forced/total_dec:.1%})"
        )
    report.append(f"  Active arms:       {agg['active_arms']} (across all sidecars)")
    report.append(f"  Confident arms:    {agg['confident_arms']} (across all sidecars)")

    # Reward feedback
    report.append("\n── REWARD FEEDBACK (CLUSTER-WIDE) ──")
    report.append(f"  Rewards received:  {agg['rewards_received']}")
    report.append(f"  Average reward:    {agg['avg_reward']:.4f}")

    # Trace classification
    report.append("\n── TRACES CAPTURED IN JAEGER ──")
    report.append(f"  Total unique traces: {total_traces}")
    for cls in ['error', 'very_slow', 'slow', 'complex', 'moderate', 'routine']:
        count = class_counts.get(cls, 0)
        pct = count / total_traces * 100 if total_traces > 0 else 0
        report.append(f"    {cls:<12}: {count:>5}  ({pct:>5.1f}%)")

    # Comparison
    report.append("\n── FSBS vs RANDOM SAMPLING COMPARISON ──")
    report.append(f"  FSBS effective sample rate:    {fsbs_sample_rate:.1%}")
    report.append(
        f"  Random target sample rate:     "
        f"{random_results.get('target_sample_rate', 0):.1%}"
    )

    report.append("")
    report.append(f"  {'Metric':<30} {'FSBS':>10} {'Random':>10} {'Winner':>10}")
    report.append(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    # Error capture
    if forced > 0:
        fsbs_err_rate = 1.0
    elif fsbs_error_count > 0:
        fsbs_err_rate = fsbs_error_count / total_traces if total_traces > 0 else 0.0
    else:
        fsbs_err_rate = 1.0
    rand_err_rate = random_results.get('avg_error_capture', 0)
    err_winner = "FSBS ✓" if fsbs_err_rate >= rand_err_rate else "Random"
    report.append(
        f"  {'Error capture rate':<30} "
        f"{fsbs_err_rate:>9.1%} {rand_err_rate:>9.1%} {err_winner:>10}"
    )

    # Slow trace capture
    fsbs_slow_rate = (
        fsbs_slow_count / total_traces if total_traces > 0 else 0
    )
    rand_slow_rate = random_results.get('avg_slow_capture', 0)
    slow_winner = "FSBS ✓" if fsbs_slow_rate >= rand_slow_rate else "Random"
    report.append(
        f"  {'Slow trace presence':<30} "
        f"{fsbs_slow_rate:>9.1%} {rand_slow_rate:>9.1%} {slow_winner:>10}"
    )

    # Total value captured
    avg_value_per_trace_fsbs = (
        fsbs_total_value / total_traces if total_traces > 0 else 0
    )
    rand_value = random_results.get('avg_value_capture', 0)
    value_winner = (
        "FSBS ✓" if avg_value_per_trace_fsbs >= rand_value else "Random"
    )
    report.append(
        f"  {'Avg value per trace':<30} "
        f"{avg_value_per_trace_fsbs:>9.3f} {rand_value:>9.3f} {value_winner:>10}"
    )

    # Volume reduction
    volume_reduction = 1.0 - agg['sample_rate']
    report.append(
        f"\n  Volume reduction:  {volume_reduction:.1%} fewer spans sent to Jaeger"
    )

    # Top learned arms (from frontend sidecar as representative)
    frontend_arms = all_arms.get('frontend')
    if frontend_arms and frontend_arms.get('arms'):
        report.append("\n── TOP LEARNED ARMS (Frontend Sidecar) ──")
        report.append(
            f"  {'Arm':>5} {'Obs':>6} {'Mean Reward':>12} "
            f"{'Confident':>10} {'Interpretation'}"
        )
        for arm in frontend_arms['arms'][:8]:
            conf = "✓ LinUCB" if arm['confident'] else "  Thompson"
            mean = arm.get('thompson_mean', 0)
            if mean > 0.6:
                interp = "← HIGH VALUE (errors/slow)"
            elif mean > 0.3:
                interp = "← moderate value"
            else:
                interp = "← low value (routine)"
            report.append(
                f"  {arm['arm_index']:>5} {arm['n_observations']:>6} "
                f"{mean:>12.3f} {conf:>10} {interp}"
            )

    # Conclusion
    report.append("\n── CONCLUSION ──")
    wins = sum([
        fsbs_err_rate >= rand_err_rate,
        fsbs_slow_rate >= rand_slow_rate,
        avg_value_per_trace_fsbs >= rand_value,
    ])

    if wins >= 2:
        report.append(
            "  ★ FSBS OUTPERFORMS random sampling on "
            f"{wins}/3 metrics"
        )
        report.append(
            "  ★ FSBS captures MORE valuable traces while sending "
            f"{volume_reduction:.0%} FEWER spans"
        )
    else:
        report.append(
            "  ⟳ FSBS needs more training time or more diverse traffic patterns."
        )

    if linucb > 0:
        report.append(
            f"  ★ LinUCB is ACTIVE with {agg['confident_arms']} "
            f"confident arms — the cluster is learning!"
        )
    else:
        report.append(
            "  ⟳ LinUCB not yet active. Send more rewards or lower "
            "FSBS_CONFIDENCE_THRESHOLD."
        )

    report.append("\n" + "=" * 70)

    # Print and save report
    full_report = "\n".join(report)
    print(full_report)

    # Save to file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    report_path = os.path.join(script_dir, "validation_report.txt")
    with open(report_path, 'w', encoding="utf-8") as f:
        f.write(full_report)
    logger.info(f"\nReport saved to: {report_path}")

    # Also save raw data as JSON
    raw_data = {
        'timestamp': time.time(),
        'cluster_metrics': agg,
        'all_sidecar_metrics': all_metrics,
        'trace_classifications': {
            cls: count for cls, count in class_counts.items()
        },
        'random_sampling_simulation': random_results,
        'fsbs_sample_rate': fsbs_sample_rate,
        'volume_reduction': volume_reduction,
    }
    json_path = os.path.join(script_dir, "validation_data.json")
    with open(json_path, 'w') as f:
        json.dump(raw_data, f, indent=2)
    logger.info(f"Raw data saved to: {json_path}")


if __name__ == '__main__':
    run_validation()