r"""
FSBS Sampler — with local checkpoint for crash recovery.

Changes from Phase 4:
  - CheckpointManager integration (save/restore state)
  - Checkpoint stats in metrics
  - restore_from_checkpoint() called on init
  - Checkpoint thread started alongside sketch updater
"""

import time
import threading
import logging
import collections
import numpy as np
from typing import Dict, Any, List

from .feature_extractor import FeatureExtractor, FeatureVector
from .count_min_sketch import CountMinSketch
from .linucb import LinUCBBandit
from .thompson import ThompsonSampler
from .mpsc_queue import MPSCQueue, SamplingRecord
from .checkpoint import CheckpointManager                      # ← CHECKPOINT

logger = logging.getLogger(__name__)


class SamplingDecision:
    """Result of a sampling decision."""
    __slots__ = ['should_sample', 'score', 'method', 'arm_index', 'feature_vector']

    def __init__(self, should_sample, score, method, arm_index, feature_vector):
        self.should_sample = should_sample
        self.score = score
        self.method = method
        self.arm_index = arm_index
        self.feature_vector = feature_vector

    def __repr__(self):
        action = "SAMPLE" if self.should_sample else "DROP"
        return (
            f"SamplingDecision({action}, score={self.score:.3f}, "
            f"method={self.method}, arm={self.arm_index})"
        )


class FSBSSampler:
    """
    Main sampler — orchestrates feature extraction, sketch, bandit, queue,
    and local checkpoint for crash recovery.
    """

    def __init__(
        self,
        service_name: str,
        alpha: float = 1.0,
        threshold: float = 0.5,
        confidence_threshold: int = 10,
        force_sample_errors: bool = True,
        checkpoint_dir: str = "",                              # ← CHECKPOINT
        checkpoint_interval: float = 60.0,                     # ← CHECKPOINT
        sketch_decay_factor: float = 0.9,          # ← ADD THIS
        sketch_decay_interval: float = 300.0,      # ← ADD THIS (seconds)
    ):
        self.service_name = service_name
        self.force_sample_errors = force_sample_errors
        self.start_time = time.time()

        # Core components
        self.feature_extractor = FeatureExtractor(service_name)
        self.sketch = CountMinSketch()
        self.bandit = LinUCBBandit(
            alpha=alpha,
            threshold=threshold,
            confidence_threshold=confidence_threshold,
        )
        self.thompson = ThompsonSampler(threshold=threshold)
        self.queue = MPSCQueue(capacity=4096)

        # Sketch decay settings
        self.sketch_decay_factor = sketch_decay_factor
        self.sketch_decay_interval = sketch_decay_interval

        # Decision metrics
        self._total_spans = 0
        self._sampled_spans = 0
        self._dropped_spans = 0
        self._thompson_decisions = 0
        self._linucb_decisions = 0
        self._forced_samples = 0

        # Reward tracking
        self._rewards_received = 0
        self._total_reward_value = 0.0
        self._reward_lock = threading.Lock()

        # Decision log
        self._decision_log = collections.deque(maxlen=2000)

        # ── Checkpoint setup ──                               # ← CHECKPOINT
        self.checkpoint_mgr = None
        if checkpoint_dir:
            self.checkpoint_mgr = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                interval_seconds=checkpoint_interval,
            )
            # Restore state from previous checkpoint if it exists
            restored = self.checkpoint_mgr.restore(
                self.sketch, self.bandit, self.thompson
            )
            if restored:
                logger.info(
                    "State restored from checkpoint — "
                    "skipping cold start!"
                )
            else:
                logger.info(
                    "No checkpoint found — starting with "
                    "Thompson sampling (cold start)"
                )

        # Background worker
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._background_worker,
            name="fsbs-sketch-updater",
            daemon=True,
        )
        self._bg_thread.start()

        # Start checkpoint background thread                   # ← CHECKPOINT
        if self.checkpoint_mgr:
            self.checkpoint_mgr.start(
                self.sketch, self.bandit, self.thompson
            )

        logger.info(
            f"FSBS Sampler initialized: service={service_name}, "
            f"alpha={alpha}, threshold={threshold}, "
            f"confidence={confidence_threshold}, "
            f"checkpoint={'enabled' if checkpoint_dir else 'disabled'}"
        )

    def decide(self, span_data: Dict[str, Any]) -> SamplingDecision:
        """Make a SAMPLE/DROP decision. HOT PATH."""
        self._total_spans += 1

        # Step 1: Extract features
        fv = self.feature_extractor.extract(span_data)

        # Step 2: Query sketch for novelty (read-only)
        novelty = self.sketch.novelty_score(fv.packed_key)
        fv.novelty_score = novelty

        # Step 3: Force-sample errors (check span_data for is_error flag)
        is_error = span_data.get('is_error', False) or fv.has_error  # ← ADD THIS
        if self.force_sample_errors and is_error:
            self._forced_samples += 1  # ← This counter now increments!
            self._sampled_spans += 1
            decision = SamplingDecision(
                should_sample=True,
                score=1.0,
                method="force_sample",
                arm_index=fv.arm_index,
                feature_vector=fv,
            )
            self._log_decision(span_data, decision)
            self._enqueue(span_data, fv, decision)
            return decision

        # Step 4: Context vector
        x = np.array(fv.to_bandit_context(), dtype=np.float32)

        # Step 5: Confidence check → LinUCB or Thompson
        if self.bandit.is_confident(fv.arm_index):
            should_sample, score = self.bandit.should_sample(fv.arm_index, x)
            method = "linucb"
            self._linucb_decisions += 1
        else:
            should_sample, score = self.thompson.should_sample(fv.arm_index)
            method = "thompson"
            self._thompson_decisions += 1

        if should_sample:
            self._sampled_spans += 1
        else:
            self._dropped_spans += 1

        decision = SamplingDecision(
            should_sample=should_sample,
            score=score,
            method=method,
            arm_index=fv.arm_index,
            feature_vector=fv,
        )

        self._log_decision(span_data, decision)
        self._enqueue(span_data, fv, decision)
        return decision

    def _log_decision(self, span_data: Dict, decision: SamplingDecision):
        """Add decision to ring buffer."""
        self._decision_log.append({
            'timestamp': time.time(),
            'trace_id': str(span_data.get('trace_id', ''))[:16],
            'service': span_data.get('service_name', ''),
            'operation': span_data.get('operation_name', ''),
            'duration_us': span_data.get('duration_us', 0),
            'arm_index': decision.arm_index,
            'method': decision.method,
            'score': round(decision.score, 4),
            'should_sample': decision.should_sample,
        })

    def _enqueue(self, span_data, fv, decision):
        """Push record to MPSC queue."""
        trace_id = span_data.get('trace_id', '')
        trace_id_hash = hash(str(trace_id)) & 0xFFFFFFFF
        record = SamplingRecord(
            trace_id_hash=trace_id_hash,
            arm_index=fv.arm_index,
            decision=1 if decision.should_sample else 0,
            feature_key=fv.packed_key,
        )
        self.queue.push(record)

    def _background_worker(self):
        """Background thread: drains queue → updates sketch, periodic decay."""
        logger.info("Background sketch updater started")
        last_decay_time = time.time()

        while self._running:
            if not self.queue.wait(timeout=1.0):
                # Even if no items, check if decay is due
                now = time.time()
                if now - last_decay_time >= self.sketch_decay_interval:
                    self.sketch.decay(self.sketch_decay_factor)
                    last_decay_time = now
                    logger.debug(
                        f"Sketch decay applied (factor={self.sketch_decay_factor})"
                    )
                continue

            records = self.queue.drain(max_items=256)
            for record in records:
                self.sketch.update(record.feature_key)

            # Check if decay is due
            now = time.time()
            if now - last_decay_time >= self.sketch_decay_interval:
                self.sketch.decay(self.sketch_decay_factor)
                last_decay_time = now
                logger.debug(
                    f"Sketch decay applied (factor={self.sketch_decay_factor})"
                )

        logger.info("Background sketch updater stopped")

    def process_reward(
        self, arm_index: int, context: List[float], reward: float
    ) -> Dict[str, Any]:
        """Process a reward signal. Called ASYNCHRONOUSLY."""
        x = np.array(context, dtype=np.float32)
        self.bandit.update(arm_index, x, reward)
        self.thompson.update(arm_index, reward)

        with self._reward_lock:
            self._rewards_received += 1
            self._total_reward_value += reward

        arm = self.bandit.arms[arm_index % self.bandit.n_arms]
        return {
            'arm_index': arm_index,
            'new_n_observations': arm.n,
            'confident': arm.n >= self.bandit.confidence_threshold,
        }

    def get_active_arms(self) -> List[Dict[str, Any]]:
        """Return stats for arms with observations."""
        active = []
        for i in range(self.bandit.n_arms):
            arm = self.bandit.arms[i]
            if arm.n > 0:
                ts = self.thompson.get_stats(i)
                active.append({
                    'arm_index': i,
                    'n_observations': arm.n,
                    'confident': arm.n >= self.bandit.confidence_threshold,
                    'thompson_mean': round(ts['mean'], 4),
                    'thompson_alpha': round(ts['alpha'], 2),
                    'thompson_beta': round(ts['beta'], 2),
                })
        return sorted(active, key=lambda x: x['n_observations'], reverse=True)

    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Return recent decisions."""
        items = list(self._decision_log)
        return items[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Full metrics snapshot."""
        total = self._total_spans
        sample_rate = self._sampled_spans / total if total > 0 else 0.0

        with self._reward_lock:
            rewards = self._rewards_received
            avg_reward = (
                self._total_reward_value / rewards if rewards > 0 else 0.0
            )

        active_arms = sum(1 for a in self.bandit.arms if a.n > 0)
        confident_arms = sum(
            1 for a in self.bandit.arms
            if a.n >= self.bandit.confidence_threshold
        )

        metrics = {
            'service': self.service_name,
            'uptime_seconds': round(time.time() - self.start_time, 1),
            'total_spans': total,
            'sampled_spans': self._sampled_spans,
            'dropped_spans': self._dropped_spans,
            'sample_rate': round(sample_rate, 4),
            'thompson_decisions': self._thompson_decisions,
            'linucb_decisions': self._linucb_decisions,
            'forced_samples': self._forced_samples,
            'rewards_received': rewards,
            'avg_reward': round(avg_reward, 4),
            'active_arms': active_arms,
            'confident_arms': confident_arms,
            'queue_size': self.queue.size,
            'queue_dropped': self.queue.dropped_count,
            'sketch_memory_bytes': self.sketch.memory_bytes,
            'bandit_memory_bytes': self.bandit.memory_bytes,
            'sketch_decay_factor': self.sketch_decay_factor,
            'sketch_decay_interval': self.sketch_decay_interval,
        }

        # Add checkpoint stats                                 # ← CHECKPOINT
        if self.checkpoint_mgr:
            metrics['checkpoint'] = self.checkpoint_mgr.get_stats()

        return metrics

    def shutdown(self):
        """Stop all threads, save final checkpoint."""
        self._running = False
        self._bg_thread.join(timeout=5.0)

        # Save final checkpoint on shutdown                    # ← CHECKPOINT
        if self.checkpoint_mgr:
            self.checkpoint_mgr.stop()

        logger.info("FSBS Sampler shut down")