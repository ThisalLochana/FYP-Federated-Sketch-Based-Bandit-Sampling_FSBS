"""
FSBS Live Dashboard — polls sidecar HTTP API, shows real-time status.

Run locally (not in Docker):
  cd D:\IIT\...\fsbs-platform
  python monitoring\dashboard.py

Requires: pip install requests (in your venv)
"""

import os
import sys
import time
import requests

SIDECAR_URL = os.environ.get('SIDECAR_URL', 'http://localhost:8081')
REFRESH_INTERVAL = 5  # seconds


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def fetch(endpoint):
    try:
        r = requests.get(f"{SIDECAR_URL}{endpoint}", timeout=3)
        return r.json()
    except Exception:
        return None


def format_bar(value, max_val, width=30):
    """Simple text progress bar."""
    if max_val == 0:
        filled = 0
    else:
        filled = int(width * value / max_val)
    return '█' * filled + '░' * (width - filled)


def render(metrics, arms_data):
    clear()

    if not metrics:
        print("  Cannot connect to FSBS sidecar at", SIDECAR_URL)
        print("  Make sure the stack is running: docker compose up -d")
        print(f"  Retrying in {REFRESH_INTERVAL}s...")
        return

    sm = metrics.get('sampler', {})
    sv = metrics.get('service', {})

    uptime = sm.get('uptime_seconds', 0)
    uptime_str = f"{int(uptime//60)}m {int(uptime%60)}s"

    total_in = sv.get('total_spans_in', 0)
    total_out = sv.get('total_spans_out', 0)
    pass_rate = total_out / total_in if total_in > 0 else 0

    thompson = sm.get('thompson_decisions', 0)
    linucb = sm.get('linucb_decisions', 0)
    forced = sm.get('forced_samples', 0)
    total_decisions = thompson + linucb + forced

    rewards = sm.get('rewards_received', 0)
    avg_rwd = sm.get('avg_reward', 0)
    active = sm.get('active_arms', 0)
    confident = sm.get('confident_arms', 0)

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║              FSBS SIDECAR — LIVE DASHBOARD              ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Uptime: {uptime_str:<12}  Threshold: {sm.get('sample_rate',0):.1%} effective   ║")
    print("  ╠══════════════════════════════════════════════════════════╣")

    print("  ║  THROUGHPUT                                             ║")
    print(f"  ║    Spans In:  {total_in:<10}  Spans Out: {total_out:<10}         ║")
    bar = format_bar(total_out, total_in, 25)
    print(f"  ║    Pass Rate: {pass_rate:.1%}  {bar}  ║")
    print(f"  ║    FWD Errors: {sv.get('forward_errors', 0):<8}                              ║")

    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  DECISION ENGINE                                        ║")

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
    print("  ║  REWARD FEEDBACK                                        ║")
    print(f"  ║    Total Rewards: {rewards:<8}  Avg Reward: {avg_rwd:.3f}           ║")
    print(f"  ║    Active Arms:   {active:<4} / 256                            ║")
    print(f"  ║    Confident:     {confident:<4} / 256  (LinUCB eligible)       ║")

    conf_bar = format_bar(confident, max(active, 1), 25)
    print(f"  ║    Confidence:    {conf_bar}  ║")

    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  TRACE CACHE                                            ║")
    print(f"  ║    Size: {sv.get('trace_cache_size',0):<8}"
          f"  Hits: {sv.get('trace_cache_hits',0):<8}"
          f"  Misses: {sv.get('trace_cache_misses',0):<8}  ║")

    # Show top arms
    if arms_data and arms_data.get('arms'):
        arms = arms_data['arms'][:6]
        print("  ╠══════════════════════════════════════════════════════════╣")
        print("  ║  TOP ARMS (by observation count)                        ║")
        for arm in arms:
            conf_marker = " ✓" if arm['confident'] else "  "
            mean = arm.get('thompson_mean', 0)
            n = arm.get('n_observations', 0)
            print(
                f"  ║    Arm {arm['arm_index']:>3}: "
                f"n={n:<5} "
                f"mean_rwd={mean:.3f} "
                f"{conf_marker}"
                f"{'  ← LinUCB' if arm['confident'] else '':<14}║"
            )

    print("  ╠══════════════════════════════════════════════════════════╣")

    if linucb > 0:
        print("  ║  ★ LinUCB is ACTIVE — the bandit is learning!           ║")
    elif rewards > 0:
        print(f"  ║  ⟳ Rewards arriving — {confident} arms nearing confidence  ║")
    else:
        print("  ║  ⏳ Waiting for reward service to start sending rewards  ║")

    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Refreshing every {REFRESH_INTERVAL}s. Press Ctrl+C to exit.")


def main():
    print(f"  FSBS Dashboard connecting to {SIDECAR_URL} ...")

    while True:
        metrics = fetch('/metrics')
        arms_data = fetch('/arms')
        render(metrics, arms_data)
        time.sleep(REFRESH_INTERVAL)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")