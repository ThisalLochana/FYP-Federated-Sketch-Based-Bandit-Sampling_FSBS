### Test Architecture
test_components.py
├── TestCountMinSketch        (8 tests)   ← Tier 2: Sketch Store
├── TestFeatureExtractor      (7 tests)   ← Tier 2: Feature Extraction
├── TestLinUCB                (7 tests)   ← Tier 2: Bandit Decision Engine
├── TestThompsonSampler       (4 tests)   ← Tier 2: Cold-Start Fallback
├── TestMPSCQueue             (3 tests)   ← Tier 2: Async Queue
├── TestFSBSSampler           (6 tests)   ← Tier 2: Full Pipeline Integration
├── TestPerformance           (4 tests)   ← Hot-Path Benchmarks
├── TestPhase4RewardIntegration (5 tests) ← Tier 4: Reward Learning
└── TestCheckpointManager     (8 tests)   ← Tier 3: Crash Recovery
                              ─────
                              52 tests

### Test Descriptions
## TestCountMinSketch (8 tests)
Tests the Count-Min Sketch probabilistic frequency estimator.
This is the component that detects novel (rare) trace patterns.

#	Test	What It Verifies
1	test_empty_sketch_returns_zero	Fresh sketch estimates 0 for any key. Confirms clean initialization of all 4×512 counters.
2	test_single_update	After inserting key once, estimate ≥ 1. Validates that all 4 hash functions update their respective row.
3	test_multiple_updates_same_key	After 100 insertions of the same key, estimate ≥ 100. Confirms counters accumulate correctly.
4	test_different_keys_no_interference	Inserting key A 1000 times does not significantly inflate key B's estimate. Validates hash independence — the minimum across 4 rows suppresses collision noise.
5	test_novelty_score_decreases_with_frequency	Novelty score for a key decreases as its frequency increases: never_seen (1.0) > seen_10_times (0.29) > seen_1000_times (0.13). This drives the bandit to explore rare patterns.
6	test_merge	Merging two sketches via element-wise maximum preserves both sketches' data. Required for distributed sketch aggregation.
7	test_memory_size	Sketch memory = exactly 8,192 bytes (4 rows × 512 cols × 4 bytes). Confirms architecture spec compliance.
8	test_serialize_deserialize	Round-trip serialization preserves all counter values. Required for checkpoint persistence and gossip (if enabled).

# Architecture mapping: Count-Min Sketch = "Sketch Store" in Tier 2 (Sidecar).
# Hot-path role: READ-ONLY query for novelty score. Writes happen on background thread.

## TestFeatureExtractor (7 tests)
# Tests the span-to-feature-vector conversion.
# This is the first component on the hot path (~3µs).

#	Test	What It Verifies
1	test_latency_buckets	Correctly maps microsecond durations to 8 exponential buckets: 0.5ms→bucket 0, 3ms→1, 10ms→2, 30ms→3, 100ms→4, 300ms→5, 1s→6, 5s→7.
2	test_error_detection	Detects errors via three methods: OTLP status_code=2 (ERROR), error=true attribute, HTTP status ≥ 500. All must set has_error=1.
3	test_service_cluster_id	Maps service names to pre-assigned 8-bit IDs: frontend=0, productcatalog=1, ..., paymentservice=5. Unknown services get ID 255.
4	test_topo_hash_different_paths	Different call paths produce different topology hashes: [frontend→checkout] ≠ [frontend→cart→checkout]. This captures trace "shape."
5	test_bandit_context_normalized	All 4 context vector elements are in [0.0, 1.0] range. Ensures numerical stability in LinUCB matrix operations.
6	test_packed_key_is_32bit	Packed feature key fits in 32-bit unsigned integer. Required for sketch hash functions.
7	test_arm_index_in_range	Arm index is always 0–255 for any service. Ensures it fits in the 256-arm bandit address space.

Architecture mapping: Feature Extractor = first hot-path component in Tier 2.
Bit layout: [topo_hash:16][svc_id:8][error:1][latency:3][reserved:4] = 32 bits.

## TestLinUCB (7 tests)
# Tests the contextual bandit that makes sampling decisions.
# This is the core ML component (~5µs per decision).

#	Test	What It Verifies
1	test_initial_arm_is_uncertain	Fresh arm has n=0 observations → not confident. Must fall back to Thompson sampling.
2	test_arm_becomes_confident_after_updates	After 10 reward updates (= confidence_threshold), arm becomes confident → LinUCB takes over from Thompson.
3	test_ucb_score_positive	UCB score is always positive because the exploration term √(xᵀA⁻¹x) > 0 when A starts as identity. Ensures the bandit always has a meaningful score.
4	test_high_reward_arm_gets_higher_score	Arm trained with reward=1.0 (50 times) gets higher UCB score than arm trained with reward=0.0 (50 times). Validates exploitation: the bandit prefers arms with higher historical rewards.
5	test_error_context_gets_higher_score	After training with "error contexts get high rewards," the bandit assigns higher UCB scores to error contexts than normal contexts. Validates contextual learning: the bandit learns feature-reward relationships, not just arm-level averages.
6	test_memory_footprint	Total bandit memory < 100KB. Architecture spec: 256 arms × 80 bytes = 20KB (we store A_inv too, so ~36KB actual).
7	test_should_sample_returns_tuple	Returns (bool, float) — not numpy types. Required for JSON serialization and Python type consistency.

# Architecture mapping: LinUCB = "LinUCB bandit" in Tier 2.
# Math: UCB = xᵀ(A⁻¹b) + α√(xᵀA⁻¹x). A⁻¹ maintained via Sherman-Morrison rank-1 updates.

## TestThompsonSampler (4 tests)
# Tests the cold-start fallback that handles arms with insufficient data.
#	Test	What It Verifies
1	test_initial_uniform_prior	With Beta(1,1) prior (uniform), sampling rate is ~50% (±15%). The bandit explores freely when it has no data.
2	test_high_reward_increases_sampling	After 100 positive rewards (α=101, β=1), sampling rate > 90%. The bandit learns to sample arms that have been valuable.
3	test_low_reward_decreases_sampling	After 100 negative rewards (α=1, β=101), sampling rate < 10%. The bandit learns to skip arms that have been worthless.
4	test_independent_arms	Rewards on arm 0 do not affect arm 1's sampling rate. Each arm's Beta distribution is independent.

# Architecture mapping: Thompson = "Thompson fallback" in Tier 2.
# Used when arm has < confidence_threshold observations. Transitions automatically to LinUCB.

## TestMPSCQueue (3 tests)
# Tests the multi-producer single-consumer queue for async sketch updates.
#	Test	What It Verifies
1	test_push_and_drain	Single item round-trip: push → drain → verify contents.
2	test_capacity_limit	Queue with maxlen=10 does not exceed 10 items. Oldest items are silently dropped (acceptable: sketch update delay, not data loss).
3	test_drain_batch_size	Drain respects max_items parameter. Prevents background thread from monopolizing CPU.

# Architecture mapping: MPSC Queue = "Async tag enqueue" in Tier 2.
# Hot-path cost: single deque.append() (~10ns in CPython due to GIL).

## TestFSBSSampler (6 tests)
# TestFSBSSampler (6 tests)
#	Test	What It Verifies
1	test_basic_decision	Pipeline returns a valid SamplingDecision for any span. End-to-end: extract → sketch → bandit → decision.
2	test_errors_always_sampled	Spans with status_code=2 (ERROR) are always sampled regardless of bandit score. method="force_sample". 100% error capture guarantee.
3	test_cold_start_uses_thompson	With confidence_threshold=100, all decisions use Thompson. Validates the cold-start path.
4	test_metrics_tracking	After 100 decisions, metrics correctly report total/sampled/dropped counts and sample_rate in [0,1].
5	test_background_worker_updates_sketch	After 50 decisions + 2-second sleep, MPSC queue is drained to 0. Validates background thread is running and consuming records.
6	test_sampling_rate_adjustable	Threshold=0.1 produces higher sampling rate than threshold=0.9. Validates that the threshold parameter controls the aggressiveness of filtering.

# Architecture mapping: FSBSSampler = "Sidecar" in Tier 2.
# Orchestrates all sub-components on the hot path.

## TestPerformance (4 tests)
# Benchmarks to verify hot-path latency budgets.

#	Test	What It Verifies	Target	Typical Result
1	test_feature_extraction_speed	Feature extraction < 100µs/call	50ns (C/Rust)	~3µs (Python)
2	test_sketch_query_speed	Sketch novelty query < 50µs/call	100ns (C/Rust)	~2µs (Python)
3	test_linucb_decision_speed	LinUCB UCB score < 100µs/call	2µs (C/Rust)	~5µs (Python)
4	test_full_pipeline_speed	Full decide() call < 1000µs/call	5µs (C/Rust)	~12µs (Python)

# Each benchmark runs 10,000 iterations after a 100-iteration warmup.
# Python overhead is ~200x the architecture target — acceptable for
# validating the algorithm; production deployment would use Rust/C++.

## TestPhase4RewardIntegration (5 tests)
# Tests the reward feedback loop that enables learning.
#	Test	What It Verifies
1	test_reward_updates_arm	Processing a reward increments the arm's observation count. After 5 rewards (with confidence_threshold=5), arm becomes confident.
2	test_linucb_activates_after_rewards	First decision uses Thompson (cold start). After 10 rewards to the same arm, subsequent decisions use LinUCB. Validates the Thompson→LinUCB transition.
3	test_decision_log	Decision ring buffer captures recent decisions with all fields (arm_index, method, score, timestamp). Used for debugging and reward correlation.
4	test_active_arms_tracking	After rewarding arms 5 and 10, get_active_arms() returns both with correct stats. Used by the monitoring dashboard.
5	test_metrics_include_rewards	After 2 rewards (0.8 and 0.6), metrics report rewards_received=2 and avg_reward=0.7.

# Architecture mapping: Tests the Tier 4 (Reward Plane) → Tier 2 (Sidecar) feedback loop.

## TestCheckpointManager (8 tests)
# Tests the local checkpoint crash recovery system.
#	Test	What It Verifies
1	test_save_creates_file	Saving creates a file on disk, size ~48KB, saves_completed increments.
2	test_restore_from_empty_dir	Restoring with no checkpoint file returns False (clean cold start). No crash, no error.
3	test_save_and_restore_preserves_sketch	Sketch counters survive save→restore cycle. Key with count=500 has count=500 after restore.
4	test_save_and_restore_preserves_linucb	LinUCB arm state survives save→restore: observation count, confidence flag, and UCB score are identical after restore.
5	test_save_and_restore_preserves_thompson	Thompson priors (α, β) survive save→restore. Arm with α=51.0, β=1.0 has same values after restore.
6	test_corrupt_file_handled_gracefully	Writing garbage to the checkpoint file does not crash the sidecar. Restore returns False, sidecar starts fresh with Thompson.
7	test_sampler_with_checkpoint_integration	Full integration: create sampler → make decisions → send rewards → save checkpoint → create NEW sampler with same checkpoint dir → verify state restored → verify LinUCB is active immediately (no cold start). This simulates a real crash recovery.
8	test_checkpoint_stats_in_metrics	Checkpoint statistics (saves_completed, checkpoint_dir) appear in the sampler's metrics output. Used by the monitoring dashboard.

# Architecture mapping: Replaces Tier 3 gossip protocol.
# File format: 16-byte header + 8,192 CMS + 37,888 LinUCB + 4,100 Thompson = ~48KB.
# Atomic write via temp file + rename prevents corruption.

## Test Design Principles
# 1. Each Component Is Tested in Isolation
CountMinSketch tests → only test sketch logic
FeatureExtractor tests → only test feature extraction
LinUCB tests → only test bandit math
Thompson tests → only test Beta sampling

# No component test depends on another component working correctly.

## 2. Integration Tests Verify the Pipeline
FSBSSampler tests → verify all components work together
  extract → sketch → bandit → decision → queue → background worker

## 3. Performance Tests Set Quantitative Bounds
Each hot-path component has a maximum latency assertion.
If a code change makes feature extraction 10x slower,
the test fails — catching regressions before deployment.

## 4. Crash Recovery Tests Simulate Real Failures
test_sampler_with_checkpoint_integration:
  1. Create sampler, train it, save checkpoint
  2. Destroy the sampler completely (simulates crash)
  3. Create NEW sampler from same checkpoint directory
  4. Verify ALL state is restored
  5. Verify LinUCB is active (no cold start)

## 5. Edge Cases and Error Handling
test_corrupt_file_handled_gracefully:
  → Garbage in checkpoint file does not crash the system
  
test_different_keys_no_interference:
  → Hash collisions do not corrupt frequency estimates
  
test_independent_arms:
  → Reward on one arm does not leak to another

## Adding New Tests
# To add a test, create a new method in the appropriate test class:
class TestCountMinSketch:
    def test_my_new_test(self):
        """Description of what this test verifies."""
        sketch = CountMinSketch()
        # ... setup ...
        assert expected == actual

# Naming convention: test_<what>_<expected_behavior>
# Run only your new test:
python -m pytest tests/test_components.py::TestCountMinSketch::test_my_new_test -v

## Coverage Map
Architecture Component          Test Class                    Coverage
─────────────────────────────────────────────────────────────────────
Feature Extractor (Tier 2)      TestFeatureExtractor          7 tests
Count-Min Sketch (Tier 2)       TestCountMinSketch            8 tests
LinUCB Bandit (Tier 2)          TestLinUCB                    7 tests
Thompson Fallback (Tier 2)      TestThompsonSampler           4 tests
MPSC Queue (Tier 2)             TestMPSCQueue                 3 tests
Sampler Pipeline (Tier 2)       TestFSBSSampler               6 tests
Hot-Path Performance (Tier 2)   TestPerformance               4 tests
Reward Integration (Tier 4)     TestPhase4RewardIntegration   5 tests
Crash Recovery (Tier 3)         TestCheckpointManager         8 tests
─────────────────────────────────────────────────────────────────────
TOTAL                                                         52 tests


---

## Quick Setup

```powershell
# Verify both files exist
type D:\IIT\4th Year\FYP\Implementation_1\fsbs-demo\fsbs-platform\README.md | Select-Object -First 3
type D:\IIT\4th Year\FYP\Implementation_1\fsbs-demo\fsbs-platform\sidecar\tests\README.md | Select-Object -First 3