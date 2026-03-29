r"""
Unit tests for all FSBS components.

Run with:
  cd C:\fsbs-demo\fsbs-platform\sidecar
  python -m pytest tests/test_components.py -v
"""

import numpy as np
import pytest
import time
import tempfile
import os
import shutil

from fsbs.count_min_sketch import CountMinSketch
from fsbs.feature_extractor import FeatureExtractor, FeatureVector
from fsbs.linucb import LinUCBBandit, LinUCBArm
from fsbs.thompson import ThompsonSampler
from fsbs.mpsc_queue import MPSCQueue, SamplingRecord
from fsbs.sampler import FSBSSampler


class TestCountMinSketch:
    """Tests for the Count-Min Sketch."""
    
    def test_empty_sketch_returns_zero(self):
        """A fresh sketch should return 0 for any key."""
        sketch = CountMinSketch()
        assert sketch.estimate(42) == 0
        assert sketch.estimate(0) == 0
        assert sketch.estimate(999999) == 0
    
    def test_single_update(self):
        """After one update, estimate should be >= 1."""
        sketch = CountMinSketch()
        sketch.update(42)
        assert sketch.estimate(42) >= 1
    
    def test_multiple_updates_same_key(self):
        """After N updates, estimate should be >= N."""
        sketch = CountMinSketch()
        for _ in range(100):
            sketch.update(42)
        assert sketch.estimate(42) >= 100
    
    def test_different_keys_no_interference(self):
        """Different keys should not significantly affect each other."""
        sketch = CountMinSketch()
        sketch.update(1, count=1000)
        # Key 2 was never inserted — its estimate should be small
        # (may be > 0 due to hash collisions, but much less than 1000)
        assert sketch.estimate(2) < 100  # generous bound for collisions
    
    def test_novelty_score_decreases_with_frequency(self):
        """More frequent patterns should have lower novelty."""
        sketch = CountMinSketch()
        
        # Novel (never seen)
        novelty_new = sketch.novelty_score(42)
        
        # Seen a few times
        for _ in range(10):
            sketch.update(42)
        novelty_medium = sketch.novelty_score(42)
        
        # Seen many times
        for _ in range(1000):
            sketch.update(42)
        novelty_common = sketch.novelty_score(42)
        
        assert novelty_new > novelty_medium > novelty_common
        assert novelty_new == pytest.approx(1.0, abs=0.01)  # brand new = max novelty
    
    def test_merge(self):
        """Merging two sketches should take element-wise maximum."""
        sketch1 = CountMinSketch()
        sketch2 = CountMinSketch()
        
        sketch1.update(1, count=100)
        sketch2.update(2, count=200)
        
        sketch1.merge(sketch2)
        
        assert sketch1.estimate(1) >= 100
        assert sketch1.estimate(2) >= 200
    
    def test_memory_size(self):
        """Sketch should be exactly 8KB."""
        sketch = CountMinSketch()
        assert sketch.memory_bytes == 8192
    
    def test_serialize_deserialize(self):
        """Round-trip serialization should preserve data."""
        sketch = CountMinSketch()
        sketch.update(42, count=100)
        sketch.update(99, count=200)
        
        data = sketch.serialize()
        restored = CountMinSketch.deserialize(data)
        
        assert restored.estimate(42) == sketch.estimate(42)
        assert restored.estimate(99) == sketch.estimate(99)


class TestFeatureExtractor:
    """Tests for the Feature Extractor."""
    
    def test_latency_buckets(self):
        """Verify latency bucket assignment."""
        extractor = FeatureExtractor("frontend")
        
        assert extractor.compute_latency_bucket(500) == 0      # 0.5ms → bucket 0
        assert extractor.compute_latency_bucket(3000) == 1      # 3ms → bucket 1
        assert extractor.compute_latency_bucket(10000) == 2     # 10ms → bucket 2
        assert extractor.compute_latency_bucket(30000) == 3     # 30ms → bucket 3
        assert extractor.compute_latency_bucket(100000) == 4    # 100ms → bucket 4
        assert extractor.compute_latency_bucket(300000) == 5    # 300ms → bucket 5
        assert extractor.compute_latency_bucket(1000000) == 6   # 1s → bucket 6
        assert extractor.compute_latency_bucket(5000000) == 7   # 5s → bucket 7
    
    def test_error_detection(self):
        """Verify error flag extraction."""
        extractor = FeatureExtractor("frontend")
        
        # No error
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        })
        assert fv.has_error == 0
        
        # Error via status code
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 1000,
            'status_code': 2,  # ERROR
            'parent_services': [],
            'attributes': {},
        })
        assert fv.has_error == 1
        
        # Error via HTTP 500
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {'http.status_code': 503},
        })
        assert fv.has_error == 1
    
    def test_service_cluster_id(self):
        """Verify service name → cluster ID mapping."""
        extractor = FeatureExtractor("frontend")
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        })
        assert fv.svc_cluster_id == 0  # frontend = ID 0
        
        fv2 = extractor.extract({
            'service_name': 'paymentservice',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        })
        assert fv2.svc_cluster_id == 5  # paymentservice = ID 5
    
    def test_topo_hash_different_paths(self):
        """Different call paths should produce different topo hashes."""
        extractor = FeatureExtractor("frontend")
        
        fv1 = extractor.extract({
            'service_name': 'checkoutservice',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': ['frontend', 'checkoutservice'],
            'attributes': {},
        })
        
        fv2 = extractor.extract({
            'service_name': 'checkoutservice',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': ['frontend', 'cartservice', 'checkoutservice'],
            'attributes': {},
        })
        
        assert fv1.topo_hash_prefix != fv2.topo_hash_prefix
    
    def test_bandit_context_normalized(self):
        """Context vector values should be in [0, 1]."""
        extractor = FeatureExtractor("frontend")
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 5000000,  # very slow
            'status_code': 2,        # error
            'parent_services': ['frontend'],
            'attributes': {},
        })
        
        ctx = fv.to_bandit_context()
        for val in ctx:
            assert 0.0 <= val <= 1.0, f"Context value {val} out of range"
    
    def test_packed_key_is_32bit(self):
        """Packed key should fit in 32 bits."""
        extractor = FeatureExtractor("frontend")
        fv = extractor.extract({
            'service_name': 'frontend',
            'duration_us': 1000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        })
        assert 0 <= fv.packed_key <= 0xFFFFFFFF
    
    def test_arm_index_in_range(self):
        """Arm index should be 0–255."""
        extractor = FeatureExtractor("frontend")
        for svc in ['frontend', 'checkoutservice', 'paymentservice', 'adservice']:
            fv = extractor.extract({
                'service_name': svc,
                'duration_us': 1000,
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            })
            assert 0 <= fv.arm_index <= 255


class TestLinUCB:
    """Tests for the LinUCB Bandit."""
    
    def test_initial_arm_is_uncertain(self):
        """Fresh arm should not be confident."""
        bandit = LinUCBBandit(confidence_threshold=10)
        assert not bandit.is_confident(0)
    
    def test_arm_becomes_confident_after_updates(self):
        """After enough updates, arm should become confident."""
        bandit = LinUCBBandit(confidence_threshold=10)
        x = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32)
        
        for _ in range(10):
            bandit.update(0, x, reward=1.0)
        
        assert bandit.is_confident(0)
    
    def test_ucb_score_positive(self):
        """UCB score should be positive (exploitation + exploration > 0)."""
        bandit = LinUCBBandit(alpha=1.0)
        x = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32)
        
        score = bandit.score(0, x)
        assert score > 0  # exploration term ensures positive score
    
    def test_high_reward_arm_gets_higher_score(self):
        """Arms with higher historical rewards should get higher UCB scores."""
        bandit = LinUCBBandit(alpha=0.1)  # low exploration to see exploitation
        x = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32)
        
        # Train arm 0 with high rewards
        for _ in range(50):
            bandit.update(0, x, reward=1.0)
        
        # Train arm 1 with low rewards
        for _ in range(50):
            bandit.update(1, x, reward=0.0)
        
        score_high = bandit.score(0, x)
        score_low = bandit.score(1, x)
        
        assert score_high > score_low
    
    def test_error_context_gets_higher_score(self):
        """
        If we train the bandit that errors are valuable,
        error contexts should get higher UCB scores.
        """
        bandit = LinUCBBandit(alpha=0.5, confidence_threshold=5)
        
        x_error = np.array([0.5, 1.0, 0.1, 0.8], dtype=np.float32)  # has_error=1
        x_normal = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32) # has_error=0
        
        # Reward errors highly on arm 0
        for _ in range(20):
            bandit.update(0, x_error, reward=1.0)
            bandit.update(0, x_normal, reward=0.1)
        
        score_error = bandit.score(0, x_error)
        score_normal = bandit.score(0, x_normal)
        
        assert score_error > score_normal
    
    def test_memory_footprint(self):
        """Check memory is in expected range."""
        bandit = LinUCBBandit(n_arms=256, d=4)
        # Should be around 20-40KB
        assert bandit.memory_bytes < 100_000  # well under 100KB
    
    def test_should_sample_returns_tuple(self):
        """should_sample should return (bool, float)."""
        bandit = LinUCBBandit()
        x = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32)
        decision, score = bandit.should_sample(0, x)
        assert isinstance(decision, bool)
        assert isinstance(score, float)


class TestThompsonSampler:
    """Tests for the Thompson Sampling Fallback."""
    
    def test_initial_uniform_prior(self):
        """With uniform prior, sampling should be roughly 50/50."""
        sampler = ThompsonSampler(threshold=0.5)
        
        # Sample 1000 times from a fresh arm
        decisions = [sampler.should_sample(0)[0] for _ in range(1000)]
        sample_rate = sum(decisions) / len(decisions)
        
        # Should be roughly 50% (±15% for randomness)
        assert 0.35 <= sample_rate <= 0.65
    
    def test_high_reward_increases_sampling(self):
        """After many positive rewards, should sample more often."""
        sampler = ThompsonSampler(threshold=0.5)
        
        # Give arm 0 many positive rewards
        for _ in range(100):
            sampler.update(0, reward=1.0)
        
        # Sample 1000 times
        decisions = [sampler.should_sample(0)[0] for _ in range(1000)]
        sample_rate = sum(decisions) / len(decisions)
        
        # Should sample almost always (>90%)
        assert sample_rate > 0.90
    
    def test_low_reward_decreases_sampling(self):
        """After many negative rewards, should drop more often."""
        sampler = ThompsonSampler(threshold=0.5)
        
        # Give arm 0 many negative rewards
        for _ in range(100):
            sampler.update(0, reward=0.0)
        
        # Sample 1000 times
        decisions = [sampler.should_sample(0)[0] for _ in range(1000)]
        sample_rate = sum(decisions) / len(decisions)
        
        # Should drop almost always (<10%)
        assert sample_rate < 0.10
    
    def test_independent_arms(self):
        """Rewards on one arm should not affect another."""
        sampler = ThompsonSampler(threshold=0.5)
        
        # Arm 0: high reward
        for _ in range(100):
            sampler.update(0, reward=1.0)
        
        # Arm 1: untouched — should still be ~50%
        decisions = [sampler.should_sample(1)[0] for _ in range(1000)]
        sample_rate = sum(decisions) / len(decisions)
        assert 0.35 <= sample_rate <= 0.65


class TestMPSCQueue:
    """Tests for the MPSC Queue."""
    
    def test_push_and_drain(self):
        """Basic push and drain."""
        queue = MPSCQueue(capacity=100)
        record = SamplingRecord(trace_id_hash=1, arm_index=0, decision=1, feature_key=42)
        
        queue.push(record)
        assert queue.size == 1
        
        items = queue.drain()
        assert len(items) == 1
        assert items[0].trace_id_hash == 1
        assert queue.size == 0
    
    def test_capacity_limit(self):
        """Queue should not exceed capacity."""
        queue = MPSCQueue(capacity=10)
        
        for i in range(20):
            queue.push(SamplingRecord(i, 0, 1, 0))
        
        # deque with maxlen drops oldest, so size stays at 10
        assert queue.size == 10
    
    def test_drain_batch_size(self):
        """Drain should respect max_items."""
        queue = MPSCQueue(capacity=100)
        
        for i in range(50):
            queue.push(SamplingRecord(i, 0, 1, 0))
        
        items = queue.drain(max_items=10)
        assert len(items) == 10
        assert queue.size == 40


class TestFSBSSampler:
    """Integration tests for the full sampler pipeline."""
    
    def test_basic_decision(self):
        """Sampler should return a decision for any span."""
        sampler = FSBSSampler(service_name="frontend")
        
        span_data = {
            'trace_id': 'abc123',
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        }
        
        decision = sampler.decide(span_data)
        assert decision.should_sample in [True, False]
        assert decision.method in ['linucb', 'thompson', 'force_sample']
        
        sampler.shutdown()
    
    def test_errors_always_sampled(self):
        """Spans with errors should always be sampled."""
        sampler = FSBSSampler(service_name="frontend", force_sample_errors=True)
        
        span_data = {
            'trace_id': 'error123',
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 2,  # ERROR
            'parent_services': [],
            'attributes': {},
        }
        
        decision = sampler.decide(span_data)
        assert decision.should_sample is True
        assert decision.method == 'force_sample'
        
        sampler.shutdown()
    
    def test_cold_start_uses_thompson(self):
        """On cold start, should use Thompson sampling."""
        sampler = FSBSSampler(
            service_name="frontend",
            confidence_threshold=100,  # very high → always cold start
        )
        
        span_data = {
            'trace_id': 'cold123',
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        }
        
        decision = sampler.decide(span_data)
        assert decision.method == 'thompson'
        
        sampler.shutdown()
    
    def test_metrics_tracking(self):
        """Metrics should be tracked correctly."""
        sampler = FSBSSampler(service_name="frontend")
        
        for i in range(100):
            span_data = {
                'trace_id': f'trace_{i}',
                'service_name': 'frontend',
                'duration_us': 5000,
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            }
            sampler.decide(span_data)
        
        metrics = sampler.get_metrics()
        assert metrics['total_spans'] == 100
        assert metrics['sampled_spans'] + metrics['dropped_spans'] == 100
        assert 0.0 <= metrics['sample_rate'] <= 1.0
        
        sampler.shutdown()
    
    def test_background_worker_updates_sketch(self):
        """Background worker should update sketch from queue."""
        sampler = FSBSSampler(service_name="frontend")
        
        # Generate some traffic
        for i in range(50):
            sampler.decide({
                'trace_id': f'trace_{i}',
                'service_name': 'frontend',
                'duration_us': 5000,
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            })
        
        # Wait for background worker to drain queue
        time.sleep(2)
        
        # Queue should be drained
        assert sampler.queue.size == 0
        
        sampler.shutdown()
    
    def test_sampling_rate_adjustable(self):
        """Different thresholds should produce different sampling rates."""
        # Low threshold → sample more
        sampler_low = FSBSSampler(service_name="frontend", threshold=0.1)
        # High threshold → sample less
        sampler_high = FSBSSampler(service_name="frontend", threshold=0.9)
        
        decisions_low = []
        decisions_high = []
        
        for i in range(200):
            span = {
                'trace_id': f'trace_{i}',
                'service_name': 'frontend',
                'duration_us': 5000 * (i % 10 + 1),  # varying latency
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            }
            decisions_low.append(sampler_low.decide(span).should_sample)
            decisions_high.append(sampler_high.decide(span).should_sample)
        
        rate_low = sum(decisions_low) / len(decisions_low)
        rate_high = sum(decisions_high) / len(decisions_high)
        
        # Low threshold should sample more than high threshold
        assert rate_low >= rate_high
        
        sampler_low.shutdown()
        sampler_high.shutdown()


class TestPerformance:
    """Performance benchmarks to verify hot-path is fast enough."""
    
    def test_feature_extraction_speed(self):
        """Feature extraction should be fast."""
        extractor = FeatureExtractor("frontend")
        span_data = {
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 0,
            'parent_services': ['frontend'],
            'attributes': {'http.method': 'GET'},
        }
        
        # Warmup
        for _ in range(100):
            extractor.extract(span_data)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            extractor.extract(span_data)
        elapsed = time.perf_counter() - start
        
        per_call_us = (elapsed / iterations) * 1_000_000
        print(f"\nFeature extraction: {per_call_us:.2f} µs/call")
        # In Python, we accept up to 50µs (architecture target is 50ns in C)
        assert per_call_us < 100  # generous bound for Python
    
    def test_sketch_query_speed(self):
        """Sketch query (hot path, read-only) should be fast."""
        sketch = CountMinSketch()
        # Pre-populate
        for i in range(10000):
            sketch.update(i)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        for i in range(iterations):
            sketch.novelty_score(i)
        elapsed = time.perf_counter() - start
        
        per_call_us = (elapsed / iterations) * 1_000_000
        print(f"\nSketch query: {per_call_us:.2f} µs/call")
        assert per_call_us < 50
    
    def test_linucb_decision_speed(self):
        """LinUCB decision should be fast."""
        bandit = LinUCBBandit()
        x = np.array([0.5, 0.0, 0.1, 0.8], dtype=np.float32)
        
        # Warmup
        for _ in range(100):
            bandit.score(0, x)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            bandit.score(0, x)
        elapsed = time.perf_counter() - start
        
        per_call_us = (elapsed / iterations) * 1_000_000
        print(f"\nLinUCB decision: {per_call_us:.2f} µs/call")
        assert per_call_us < 100
    
    def test_full_pipeline_speed(self):
        """Full decide() call should be under 1ms in Python."""
        sampler = FSBSSampler(service_name="frontend")
        span_data = {
            'trace_id': 'perf_test',
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 0,
            'parent_services': ['frontend'],
            'attributes': {'http.method': 'GET'},
        }
        
        # Warmup
        for _ in range(100):
            sampler.decide(span_data)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 10000
        for _ in range(iterations):
            sampler.decide(span_data)
        elapsed = time.perf_counter() - start
        
        per_call_us = (elapsed / iterations) * 1_000_000
        print(f"\nFull pipeline: {per_call_us:.2f} µs/call")
        # Python target: under 1000µs (1ms)
        # Architecture target in C/Rust: under 5µs
        assert per_call_us < 1000
        
        sampler.shutdown()

class TestPhase4RewardIntegration:
    """Tests for Phase 4 reward processing."""

    def test_reward_updates_arm(self):
        """Processing a reward should update the arm's observation count."""
        sampler = FSBSSampler(service_name="frontend", confidence_threshold=5)

        context = [0.5, 1.0, 0.0, 0.8]
        result = sampler.process_reward(arm_index=5, context=context, reward=1.0)

        assert result['arm_index'] == 5
        assert result['new_n_observations'] == 1
        assert result['confident'] is False  # need 5 observations

        # Send 4 more rewards
        for _ in range(4):
            result = sampler.process_reward(5, context, 0.8)

        assert result['new_n_observations'] == 5
        assert result['confident'] is True  # now confident!

        sampler.shutdown()

    def test_linucb_activates_after_rewards(self):
        """After enough rewards, decisions should use LinUCB."""
        sampler = FSBSSampler(
            service_name="frontend",
            confidence_threshold=5,
        )

        # First decision should use Thompson (cold start)
        span = {
            'trace_id': 'test1',
            'service_name': 'frontend',
            'duration_us': 5000,
            'status_code': 0,
            'parent_services': [],
            'attributes': {},
        }
        d1 = sampler.decide(span)
        assert d1.method == 'thompson'

        # Send rewards to the same arm
        context = d1.feature_vector.to_bandit_context()
        arm_idx = d1.arm_index
        for _ in range(10):
            sampler.process_reward(arm_idx, context, 0.5)

        # Next decision for same arm should use LinUCB
        span['trace_id'] = 'test2'
        d2 = sampler.decide(span)
        assert d2.method == 'linucb'

        sampler.shutdown()

    def test_decision_log(self):
        """Decision log should capture recent decisions."""
        sampler = FSBSSampler(service_name="frontend")

        for i in range(10):
            sampler.decide({
                'trace_id': f'trace_{i}',
                'service_name': 'frontend',
                'duration_us': 5000,
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            })

        recent = sampler.get_recent_decisions(limit=5)
        assert len(recent) == 5
        assert 'arm_index' in recent[0]
        assert 'method' in recent[0]
        assert 'score' in recent[0]

        sampler.shutdown()

    def test_active_arms_tracking(self):
        """Active arms should be trackable after rewards."""
        sampler = FSBSSampler(service_name="frontend")

        # Initially no active arms
        arms = sampler.get_active_arms()
        assert len(arms) == 0

        # Send rewards to two arms
        sampler.process_reward(5, [0.5, 0.0, 0.1, 0.8], 1.0)
        sampler.process_reward(10, [0.3, 1.0, 0.2, 0.5], 0.5)

        arms = sampler.get_active_arms()
        assert len(arms) == 2
        assert arms[0]['arm_index'] in [5, 10]

        sampler.shutdown()

    def test_metrics_include_rewards(self):
        """Metrics should include reward tracking."""
        sampler = FSBSSampler(service_name="frontend")

        sampler.process_reward(0, [0.5, 0.0, 0.1, 0.8], 0.8)
        sampler.process_reward(0, [0.5, 0.0, 0.1, 0.8], 0.6)

        metrics = sampler.get_metrics()
        assert metrics['rewards_received'] == 2
        assert metrics['avg_reward'] == pytest.approx(0.7, abs=0.01)

        sampler.shutdown()

class TestCheckpointManager:
    """Tests for the local checkpoint crash recovery system."""

    def _make_temp_dir(self):
        """Create a temporary directory for checkpoint files."""
        return tempfile.mkdtemp(prefix='fsbs_test_ckpt_')

    def test_save_creates_file(self):
        """Saving a checkpoint should create a file on disk."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            sketch = CountMinSketch()
            bandit = LinUCBBandit()
            thompson = ThompsonSampler()

            result = mgr.save(sketch, bandit, thompson)
            assert result is True
            assert os.path.exists(mgr.filepath)
            assert mgr.saves_completed == 1

            # File should be ~48KB
            size = os.path.getsize(mgr.filepath)
            assert 40_000 < size < 60_000, f"Unexpected size: {size}"
        finally:
            shutil.rmtree(ckpt_dir)

    def test_restore_from_empty_dir(self):
        """Restoring with no checkpoint file should return False."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            sketch = CountMinSketch()
            bandit = LinUCBBandit()
            thompson = ThompsonSampler()

            result = mgr.restore(sketch, bandit, thompson)
            assert result is False
            assert mgr.restore_success is False
        finally:
            shutil.rmtree(ckpt_dir)

    def test_save_and_restore_preserves_sketch(self):
        """Sketch counters should survive save→restore cycle."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            # Create and populate sketch
            sketch = CountMinSketch()
            sketch.update(42, count=500)
            sketch.update(99, count=200)
            original_42 = sketch.estimate(42)
            original_99 = sketch.estimate(99)

            bandit = LinUCBBandit()
            thompson = ThompsonSampler()

            # Save
            mgr.save(sketch, bandit, thompson)

            # Create fresh components
            new_sketch = CountMinSketch()
            new_bandit = LinUCBBandit()
            new_thompson = ThompsonSampler()

            assert new_sketch.estimate(42) == 0  # fresh = empty

            # Restore
            result = mgr.restore(new_sketch, new_bandit, new_thompson)
            assert result is True
            assert new_sketch.estimate(42) == original_42
            assert new_sketch.estimate(99) == original_99
        finally:
            shutil.rmtree(ckpt_dir)

    def test_save_and_restore_preserves_linucb(self):
        """LinUCB arm state should survive save→restore cycle."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            sketch = CountMinSketch()
            bandit = LinUCBBandit(confidence_threshold=5)
            thompson = ThompsonSampler()

            # Train arm 7 with 20 observations
            x = np.array([0.5, 1.0, 0.1, 0.8], dtype=np.float32)
            for _ in range(20):
                bandit.update(7, x, reward=0.9)

            assert bandit.arms[7].n == 20
            assert bandit.is_confident(7) is True
            original_score = bandit.score(7, x)

            # Save
            mgr.save(sketch, bandit, thompson)

            # Create fresh components
            new_bandit = LinUCBBandit(confidence_threshold=5)
            assert new_bandit.arms[7].n == 0  # fresh = empty

            # Restore
            mgr.restore(CountMinSketch(), new_bandit, ThompsonSampler())

            assert new_bandit.arms[7].n == 20
            assert new_bandit.is_confident(7) is True
            restored_score = new_bandit.score(7, x)
            assert restored_score == pytest.approx(original_score, abs=0.01)
        finally:
            shutil.rmtree(ckpt_dir)

    def test_save_and_restore_preserves_thompson(self):
        """Thompson priors should survive save→restore cycle."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            sketch = CountMinSketch()
            bandit = LinUCBBandit()
            thompson = ThompsonSampler()

            # Give arm 3 many positive rewards
            for _ in range(50):
                thompson.update(3, reward=1.0)

            original_stats = thompson.get_stats(3)
            assert original_stats['alpha'] == pytest.approx(51.0)

            # Save
            mgr.save(sketch, bandit, thompson)

            # Create fresh and restore
            new_thompson = ThompsonSampler()
            mgr.restore(CountMinSketch(), LinUCBBandit(), new_thompson)

            restored_stats = new_thompson.get_stats(3)
            assert restored_stats['alpha'] == pytest.approx(51.0)
            assert restored_stats['beta'] == pytest.approx(1.0)
        finally:
            shutil.rmtree(ckpt_dir)

    def test_corrupt_file_handled_gracefully(self):
        """Corrupted checkpoint should be handled without crash."""
        ckpt_dir = self._make_temp_dir()
        try:
            from fsbs.checkpoint import CheckpointManager
            mgr = CheckpointManager(checkpoint_dir=ckpt_dir)

            # Write garbage to the checkpoint file
            with open(mgr.filepath, 'wb') as f:
                f.write(b'GARBAGE DATA THAT IS NOT A VALID CHECKPOINT')

            sketch = CountMinSketch()
            bandit = LinUCBBandit()
            thompson = ThompsonSampler()

            # Should return False and not crash
            result = mgr.restore(sketch, bandit, thompson)
            assert result is False
        finally:
            shutil.rmtree(ckpt_dir)

    def test_sampler_with_checkpoint_integration(self):
        """Full sampler should work with checkpoint enabled."""
        ckpt_dir = self._make_temp_dir()
        try:
            # Create sampler with checkpoint
            sampler = FSBSSampler(
                service_name="frontend",
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=999,  # don't auto-save during test
            )

            # Make some decisions and send rewards
            for i in range(20):
                sampler.decide({
                    'trace_id': f'trace_{i}',
                    'service_name': 'frontend',
                    'duration_us': 5000,
                    'status_code': 0,
                    'parent_services': [],
                    'attributes': {},
                })

            context = [0.5, 0.0, 0.0, 0.8]
            for _ in range(15):
                sampler.process_reward(0, context, 0.7)

            assert sampler.bandit.arms[0].n == 15

            # Manually trigger checkpoint save
            sampler.checkpoint_mgr.save(
                sampler.sketch, sampler.bandit, sampler.thompson
            )

            sampler.shutdown()

            # Create NEW sampler with same checkpoint dir
            # This simulates a crash recovery
            sampler2 = FSBSSampler(
                service_name="frontend",
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=999,
            )

            # Verify state was restored
            assert sampler2.bandit.arms[0].n == 15
            assert sampler2.bandit.is_confident(0) is True

            # New decisions should use LinUCB (not Thompson)
            decision = sampler2.decide({
                'trace_id': 'recovery_test',
                'service_name': 'frontend',
                'duration_us': 5000,
                'status_code': 0,
                'parent_services': [],
                'attributes': {},
            })
            assert decision.method == 'linucb'

            sampler2.shutdown()
        finally:
            shutil.rmtree(ckpt_dir)

    def test_checkpoint_stats_in_metrics(self):
        """Checkpoint stats should appear in sampler metrics."""
        ckpt_dir = self._make_temp_dir()
        try:
            sampler = FSBSSampler(
                service_name="frontend",
                checkpoint_dir=ckpt_dir,
                checkpoint_interval=999,
            )

            metrics = sampler.get_metrics()
            assert 'checkpoint' in metrics
            assert metrics['checkpoint']['saves_completed'] == 0
            assert metrics['checkpoint']['checkpoint_dir'] == ckpt_dir

            sampler.shutdown()
        finally:
            shutil.rmtree(ckpt_dir)