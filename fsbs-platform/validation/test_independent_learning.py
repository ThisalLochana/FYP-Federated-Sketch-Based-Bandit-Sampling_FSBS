"""
Test that independent sidecars learn the same arm weights.

This proves that no shared checkpoint is needed — each sidecar
converges to the same policy because they see the same traffic patterns.

Usage:
  1. Deploy the per-pod sidecar architecture
  2. Run traffic for 20-30 minutes
  3. Run this script to compare arm weights across sidecars

  python validation\test_independent_learning.py
"""

import requests
import json

SIDECARS = {
    'frontend': 'http://localhost:8081',
    'productcatalog': 'http://localhost:8082',
    'cart': 'http://localhost:8083',
    'currency': 'http://localhost:8084',
    'checkout': 'http://localhost:8085',
    'payment': 'http://localhost:8086',
    'shipping': 'http://localhost:8087',
    'email': 'http://localhost:8088',
    'recommendation': 'http://localhost:8089',
    'ad': 'http://localhost:8090',
    'loadgen': 'http://localhost:8091',
}


def fetch_arms(url):
    """Fetch arm statistics from a sidecar."""
    try:
        r = requests.get(f"{url}/arms", timeout=3)
        return r.json()
    except Exception as e:
        print(f"  Error fetching from {url}: {e}")
        return None


def main():
    print("=" * 70)
    print("INDEPENDENT LEARNING TEST")
    print("Comparing arm weights across all sidecars")
    print("=" * 70)

    all_arms_data = {}
    
    for name, url in SIDECARS.items():
        print(f"\nFetching arms from {name} sidecar ({url})...")
        arms_data = fetch_arms(url)
        if arms_data:
            all_arms_data[name] = arms_data
            print(f"  ✓ {arms_data['active_arms']} active arms, "
                  f"{arms_data['confident_arms']} confident")
        else:
            print(f"  ✗ Failed to fetch (sidecar may be down)")

    if not all_arms_data:
        print("\nNo sidecars reachable. Make sure the stack is running:")
        print("  docker compose up -d")
        return

    # Build a map: arm_index → list of (sidecar_name, arm_data)
    arm_index_map = {}
    for sidecar_name, data in all_arms_data.items():
        for arm in data['arms']:
            idx = arm['arm_index']
            if idx not in arm_index_map:
                arm_index_map[idx] = []
            arm_index_map[idx].append((sidecar_name, arm))

    # Find arms that appear in multiple sidecars
    shared_arms = {
        idx: observations 
        for idx, observations in arm_index_map.items() 
        if len(observations) > 1
    }

    if not shared_arms:
        print("\n" + "=" * 70)
        print("RESULT: No arms are shared across multiple sidecars")
        print("This is normal if traffic has been low or very service-specific.")
        print("=" * 70)
        return

    print("\n" + "=" * 70)
    print("ARMS THAT APPEAR IN MULTIPLE SIDECARS")
    print("=" * 70)

    for arm_idx, observations in sorted(shared_arms.items()):
        print(f"\n── Arm {arm_idx} ──")
        means = []
        for sidecar_name, arm_data in observations:
            mean = arm_data['thompson_mean']
            n = arm_data['n_observations']
            means.append(mean)
            conf = "✓" if arm_data['confident'] else " "
            print(f"  {conf} {sidecar_name:<15}: "
                  f"n={n:<5} mean_reward={mean:.3f}")
        
        # Calculate variance across sidecars
        avg_mean = sum(means) / len(means)
        variance = sum((m - avg_mean) ** 2 for m in means) / len(means)
        std_dev = variance ** 0.5
        
        print(f"  Average mean: {avg_mean:.3f}  Std dev: {std_dev:.4f}")
        
        if std_dev < 0.05:
            print(f"  ✓ CONVERGENCE: All sidecars learned similar weights")
        else:
            print(f"  ⚠ DIVERGENCE: Sidecars learned different weights (may need more data)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    converged_arms = sum(
        1 for observations in shared_arms.values()
        if (sum(arm['thompson_mean'] for _, arm in observations) / len(observations))
           and all(abs(arm['thompson_mean'] - sum(a['thompson_mean'] for _, a in observations) / len(observations)) < 0.05
                   for _, arm in observations)
    )
    
    total_shared = len(shared_arms)
    
    if converged_arms == total_shared:
        print(f"✓ ALL {total_shared} shared arms converged to similar weights")
        print("  Independent learning works — no shared checkpoint needed!")
    elif converged_arms > total_shared * 0.7:
        print(f"⟳ {converged_arms}/{total_shared} shared arms converged")
        print("  Mostly working — may need more training time or traffic")
    else:
        print(f"✗ Only {converged_arms}/{total_shared} shared arms converged")
        print("  Check if reward service is running and sending feedback")
    
    print("=" * 70)


if __name__ == '__main__':
    main()