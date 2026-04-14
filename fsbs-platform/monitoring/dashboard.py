"""
FSBS Live Dashboard — polls all sidecar HTTP APIs, shows aggregated real-time status.

Run locally (not in Docker):
  cd D:\IIT\...\fsbs-platform
  python monitoring\dashboard.py

Requires: pip install requests (in your venv)
"""

import os
import sys
import time
import requests

SIDECAR_URLS = [
    'http://localhost:8081',  # frontend
    'http://localhost:8082',  # productcatalog
    'http://localhost:8083',  # cart
    'http://localhost:8084',  # currency
    'http://localhost:8085',  # checkout
    'http://localhost:8086',  # payment
    'http://localhost:8087',  # shipping
    'http://localhost:8088',  # email
    'http://localhost:8089',  # recommendation
    'http://localhost:8090',  # ad
    'http://localhost:8091',  # loadgen
]

SIDECAR_NAMES = [
    'frontend',
    'productcatalog',
    'cart',
    'currency',
    'checkout',
    'payment',
    'shipping',
    'email',
    'recommendation',
    'ad',
    'loadgen',
]

REFRESH_INTERVAL = 5  # seconds


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def fetch_all_metrics():
    """Fetch metrics from all sidecars."""
    all_metrics = []
    for url in SIDECAR_URLS:
        try:
            r = requests.get(f"{url}/metrics", timeout=2)
            all_metrics.append(r.json())
        except Exception:
            all_metrics.append(None)
    return all_metrics


def fetch_all_arms():
    """Fetch arm stats from all sidecars."""
    all_arms = []
    for url in SIDECAR_URLS:
        try:
            r = requests.get(f"{url}/arms", timeout=2)
            all_arms.append(r.json())
        except Exception:
            all_arms.append(None)
    return all_arms


def aggregate_metrics(all_metrics):
    """Combine metrics from all sidecars into cluster-wide totals."""
    valid = [m for m in all_metrics if m is not None]
    
    if not valid:
        return None
    
    # Sum up all the counters
    total_spans_in = sum(m['service']['total_spans_in'] for m in valid)
    total_spans_out = sum(m['service']['total_spans_out'] for m in valid)
    thompson = sum(m['sampler']['thompson_decisions'] for m in valid)
    linucb = sum(m['sampler']['linucb_decisions'] for m in valid)
    forced = sum(m['sampler']['forced_samples'] for m in valid)
    rewards = sum(m['sampler']['rewards_received'] for m in valid)
    
    # Average uptime across all sidecars
    avg_uptime = sum(m['sampler']['uptime_seconds'] for m in valid) / len(valid)
    
    # Weighted average reward (by number of rewards)
    total_reward_value = sum(
        m['sampler']['rewards_received'] * m['sampler']['avg_reward'] 
        for m in valid
    )
    avg_reward = total_reward_value / rewards if rewards > 0 else 0.0
    
    # Count unique active/confident arms across all sidecars
    all_active_arms = set()
    all_confident_arms = set()
    for m in valid:
        all_active_arms.add(m['sampler']['active_arms'])
        all_confident_arms.add(m['sampler']['confident_arms'])
    
    total_active = sum(m['sampler']['active_arms'] for m in valid)
    total_confident = sum(m['sampler']['confident_arms'] for m in valid)
    
    return {
        'total_spans_in': total_spans_in,
        'total_spans_out': total_spans_out,
        'sample_rate': total_spans_out / total_spans_in if total_spans_in > 0 else 0,
        'thompson': thompson,
        'linucb': linucb,
        'forced': forced,
        'rewards': rewards,
        'avg_reward': avg_reward,
        'avg_uptime': avg_uptime,
        'total_active_arms': total_active,
        'total_confident_arms': total_confident,
        'num_sidecars_alive': len(valid),
        'num_sidecars_total': len(all_metrics),
    }


def format_bar(value, max_val, width=30):
    """Simple text progress bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = int(width * value / max_val)
    return '█' * filled + '░' * (width - filled)


def render(agg, all_metrics, all_arms):
    clear()

    if not agg:
        print("  Cannot connect to any FSBS sidecars")
        print(f"  Trying: {', '.join(SIDECAR_URLS[:3])}...")
        print("  Make sure the stack is running: docker compose up -d")
        print(f"  Retrying in {REFRESH_INTERVAL}s...")
        return

    uptime = agg['avg_uptime']
    uptime_str = f"{int(uptime//60)}m {int(uptime%60)}s"

    total_in = agg['total_spans_in']
    total_out = agg['total_spans_out']
    pass_rate = agg['sample_rate']

    thompson = agg['thompson']
    linucb = agg['linucb']
    forced = agg['forced']
    total_decisions = thompson + linucb + forced

    rewards = agg['rewards']
    avg_rwd = agg['avg_reward']

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║         FSBS CLUSTER — LIVE DASHBOARD (ALL SIDECARS)    ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Sidecars: {agg['num_sidecars_alive']}/{agg['num_sidecars_total']} alive     "
          f"Avg Uptime: {uptime_str:<12}        ║")
    print("  ╠══════════════════════════════════════════════════════════╣")

    print("  ║  CLUSTER THROUGHPUT                                     ║")
    print(f"  ║    Spans In:  {total_in:<10}  Spans Out: {total_out:<10}         ║")
    bar = format_bar(total_out, total_in, 25)
    print(f"  ║    Pass Rate: {pass_rate:.1%}  {bar}  ║")

    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  DECISION ENGINE (CLUSTER-WIDE)                         ║")

    if total_decisions > 0:
        t_pct = thompson / total_decisions * 100
        l_pct = linucb / total_decisions * 100
        f_pct = forced / total_decisions * 100
    else:
        t_pct = l_pct = f_pct = 0

    t_status = "← cold start" if linucb == 0 else ""
    l_status = "← LEARNING ACTIVE!" if linucb > 0 else "← waiting for rewards"

    print(f"  ║    Thompson: {thompson:<8} ({t_pct:5.1f}%)  {t_status:<20} ║")
    print(f"  ║    LinUCB:   {linucb:<8} ({l_pct:5.1f}%)  {l_status:<20} ║")
    print(f"  ║    Forced:   {forced:<8} ({f_pct:5.1f}%)  (error traces)        ║")

    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  REWARD FEEDBACK (CLUSTER-WIDE)                         ║")
    print(f"  ║    Total Rewards: {rewards:<8}  Avg Reward: {avg_rwd:.3f}           ║")
    print(f"  ║    Active Arms:   {agg['total_active_arms']:<4} (across all sidecars)           ║")
    print(f"  ║    Confident:     {agg['total_confident_arms']:<4} (across all sidecars)           ║")

    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  PER-SIDECAR STATUS                                     ║")
    
    for i, (name, metrics) in enumerate(zip(SIDECAR_NAMES, all_metrics)):
        if metrics is None:
            status = "✗ DOWN"
            details = ""
        else:
            sm = metrics['sampler']
            sv = metrics['service']
            total_in_sidecar = sv['total_spans_in']
            total_out_sidecar = sv['total_spans_out']
            rate = total_out_sidecar / total_in_sidecar if total_in_sidecar > 0 else 0
            status = "✓"
            details = f"in={total_in_sidecar:<6} out={total_out_sidecar:<6} rate={rate:.1%}"
        
        print(f"  ║  {status} {name:<15} {details:<30} ║")

    print("  ╠══════════════════════════════════════════════════════════╣")

    # Show top arms across ALL sidecars (merged view)
    merged_arms = {}
    for arms_data in all_arms:
        if arms_data and arms_data.get('arms'):
            for arm in arms_data['arms']:
                idx = arm['arm_index']
                if idx not in merged_arms:
                    merged_arms[idx] = {
                        'n_observations': 0,
                        'total_alpha': 0,
                        'total_beta': 0,
                        'count': 0,
                    }
                merged_arms[idx]['n_observations'] += arm['n_observations']
                merged_arms[idx]['total_alpha'] += arm['thompson_alpha']
                merged_arms[idx]['total_beta'] += arm['thompson_beta']
                merged_arms[idx]['count'] += 1
    
    # Calculate average thompson_mean for each arm
    for idx, data in merged_arms.items():
        alpha_avg = data['total_alpha'] / data['count']
        beta_avg = data['total_beta'] / data['count']
        data['mean'] = alpha_avg / (alpha_avg + beta_avg)
        data['confident'] = data['n_observations'] >= 10
    
    # Sort by total observations
    top_arms = sorted(
        merged_arms.items(),
        key=lambda x: x[1]['n_observations'],
        reverse=True
    )[:6]
    
    if top_arms:
        print("  ║  TOP ARMS (merged across all sidecars)                  ║")
        for arm_idx, data in top_arms:
            conf_marker = " ✓" if data['confident'] else "  "
            mean = data['mean']
            n = data['n_observations']
            print(
                f"  ║    Arm {arm_idx:>3}: "
                f"n={n:<5} "
                f"mean_rwd={mean:.3f} "
                f"{conf_marker}"
                f"{'  ← LinUCB' if data['confident'] else '':<14}║"
            )

    print("  ╠══════════════════════════════════════════════════════════╣")

    if linucb > 0:
        print("  ║  ★ LinUCB is ACTIVE — the cluster is learning!          ║")
    elif rewards > 0:
        print(f"  ║  ⟳ Rewards arriving — arms nearing confidence           ║")
    else:
        print("  ║  ⏳ Waiting for reward service to start sending rewards  ║")

    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Refreshing every {REFRESH_INTERVAL}s. Press Ctrl+C to exit.")


def main():
    print(f"  FSBS Cluster Dashboard")
    print(f"  Monitoring {len(SIDECAR_URLS)} sidecars...")
    print()

    while True:
        all_metrics = fetch_all_metrics()
        all_arms = fetch_all_arms()
        agg = aggregate_metrics(all_metrics)
        render(agg, all_metrics, all_arms)
        time.sleep(REFRESH_INTERVAL)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")