r"""
Local Checkpoint — crash recovery via periodic state persistence.

Replaces gossip protocol with a simpler, zero-network-overhead approach.

Every 60 seconds, a background thread serializes the full sidecar state
(~48KB) to a local file using atomic write (temp file + rename).

On startup, if a checkpoint file exists, the sidecar restores its state
and resumes from where it left off — no cold start, no Thompson fallback.

File format (binary):
  Header (16 bytes):
    Magic:     4 bytes  "FSBS"
    Version:   4 bytes  uint32 = 1
    Timestamp: 8 bytes  float64 (unix time)

  CMS section:
    Length:    4 bytes
    Data:      8,192 bytes (4 rows × 512 cols × 4 bytes)

  LinUCB section:
    n_arms:   4 bytes  uint32
    d:        4 bytes  uint32
    Per arm (256 arms):
      n:      4 bytes  uint32
      A:      64 bytes (4×4 float32)
      A_inv:  64 bytes (4×4 float32)
      b:      16 bytes (4×1 float32)
    Total:    256 × 148 = 37,888 bytes

  Thompson section:
    n_arms:   4 bytes  uint32
    alphas:   2,048 bytes (256 × float64)
    betas:    2,048 bytes (256 × float64)

  Total file size: ~48KB
  Write time: <1ms (SSD) to ~5ms (HDD)
  Zero network traffic.
"""

import os
import struct
import time
import logging
import threading
import tempfile
import shutil
import numpy as np
from typing import Optional

from .count_min_sketch import CountMinSketch
from .linucb import LinUCBBandit
from .thompson import ThompsonSampler

logger = logging.getLogger(__name__)

# File format constants
CHECKPOINT_MAGIC = b'FSBS'
CHECKPOINT_VERSION = 1


class CheckpointManager:
    """
    Manages periodic saving and loading of sidecar state.

    Usage:
        mgr = CheckpointManager(
            checkpoint_dir="/data/checkpoints",
            interval_seconds=60,
        )

        # On startup — restore if checkpoint exists
        mgr.restore(sketch, bandit, thompson)

        # Start background saving
        mgr.start(sketch, bandit, thompson)

        # On shutdown
        mgr.stop()
    """

    def __init__(
        self,
        checkpoint_dir: str = "/data/checkpoints",
        interval_seconds: float = 60.0,
        filename: str = "fsbs_state.ckpt",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.interval = interval_seconds
        self.filepath = os.path.join(checkpoint_dir, filename)

        # Ensure directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Background thread state
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.saves_completed = 0
        self.saves_failed = 0
        self.last_save_time: Optional[float] = None
        self.last_save_size: int = 0
        self.restore_success = False
        self.restore_age_seconds: float = 0.0

    # ── Serialization ─────────────────────────────────────────

    def save(
        self,
        sketch: CountMinSketch,
        bandit: LinUCBBandit,
        thompson: ThompsonSampler,
    ) -> bool:
        """
        Save full sidecar state to checkpoint file.
        Uses atomic write: write to temp file, then rename.
        This prevents corruption if the process crashes mid-write.

        Returns True if successful.
        """
        try:
            data = bytearray()

            # ── Header ──
            data.extend(CHECKPOINT_MAGIC)
            data.extend(struct.pack('<I', CHECKPOINT_VERSION))
            data.extend(struct.pack('<d', time.time()))

            # ── CMS section ──
            cms_data = sketch.serialize()
            data.extend(struct.pack('<I', len(cms_data)))
            data.extend(cms_data)

            # ── LinUCB section ──
            linucb_data = self._serialize_linucb(bandit)
            data.extend(linucb_data)

            # ── Thompson section ──
            thompson_data = self._serialize_thompson(thompson)
            data.extend(thompson_data)

            # ── Atomic write ──
            # Write to temp file in the same directory (same filesystem)
            # then rename — rename is atomic on most filesystems
            fd, tmp_path = tempfile.mkstemp(
                dir=self.checkpoint_dir,
                prefix='fsbs_tmp_',
                suffix='.ckpt',
            )
            try:
                os.write(fd, bytes(data))
                os.fsync(fd)  # flush to disk
                os.close(fd)

                # Atomic rename (overwrites existing checkpoint)
                # On Windows, need to remove target first
                if os.path.exists(self.filepath):
                    os.replace(tmp_path, self.filepath)
                else:
                    os.rename(tmp_path, self.filepath)

            except Exception:
                os.close(fd)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

            self.saves_completed += 1
            self.last_save_time = time.time()
            self.last_save_size = len(data)

            logger.debug(
                f"Checkpoint saved: {len(data)} bytes "
                f"(save #{self.saves_completed})"
            )
            return True

        except Exception as e:
            self.saves_failed += 1
            logger.error(f"Checkpoint save failed: {e}")
            return False

    def restore(
        self,
        sketch: CountMinSketch,
        bandit: LinUCBBandit,
        thompson: ThompsonSampler,
    ) -> bool:
        """
        Restore sidecar state from checkpoint file.
        Returns True if successfully restored.
        Returns False if no checkpoint exists or restoration fails
        (sidecar continues with fresh state — Thompson fallback).
        """
        if not os.path.exists(self.filepath):
            logger.info("No checkpoint file found — starting fresh")
            return False

        try:
            with open(self.filepath, 'rb') as f:
                data = f.read()

            offset = 0

            # ── Header ──
            magic = data[offset:offset + 4]
            offset += 4
            if magic != CHECKPOINT_MAGIC:
                logger.warning(
                    f"Invalid checkpoint magic: {magic} — starting fresh"
                )
                return False

            version = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            if version != CHECKPOINT_VERSION:
                logger.warning(
                    f"Checkpoint version mismatch: {version} vs "
                    f"{CHECKPOINT_VERSION} — starting fresh"
                )
                return False

            timestamp = struct.unpack('<d', data[offset:offset + 8])[0]
            offset += 8
            age = time.time() - timestamp
            self.restore_age_seconds = age

            # ── CMS section ──
            cms_length = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4
            cms_data = data[offset:offset + cms_length]
            offset += cms_length

            restored_sketch = CountMinSketch.deserialize(cms_data)
            # Copy restored data into the existing sketch
            sketch._counters = restored_sketch._counters

            # ── LinUCB section ──
            offset = self._deserialize_linucb(data, offset, bandit)

            # ── Thompson section ──
            offset = self._deserialize_thompson(data, offset, thompson)

            self.restore_success = True
            logger.info(
                f"Checkpoint restored: age={age:.1f}s, "
                f"size={len(data)} bytes"
            )
            return True

        except Exception as e:
            logger.error(
                f"Checkpoint restore failed: {e} — starting fresh"
            )
            self.restore_success = False
            return False

    # ── LinUCB serialization ──────────────────────────────────

    def _serialize_linucb(self, bandit: LinUCBBandit) -> bytes:
        """Serialize all LinUCB arms to bytes."""
        data = bytearray()
        data.extend(struct.pack('<I', bandit.n_arms))
        data.extend(struct.pack('<I', bandit.d))

        for arm in bandit.arms:
            data.extend(struct.pack('<I', arm.n))
            data.extend(arm.A.astype(np.float32).tobytes())
            data.extend(arm.A_inv.astype(np.float32).tobytes())
            data.extend(arm.b.astype(np.float32).tobytes())

        return bytes(data)

    def _deserialize_linucb(
        self, data: bytes, offset: int, bandit: LinUCBBandit
    ) -> int:
        """Deserialize LinUCB arms from bytes. Returns new offset."""
        n_arms = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        d = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        if n_arms != bandit.n_arms or d != bandit.d:
            logger.warning(
                f"LinUCB dimension mismatch: file has {n_arms} arms, "
                f"d={d} but bandit has {bandit.n_arms} arms, d={bandit.d}"
            )
            # Skip past the data
            per_arm = 4 + d * d * 4 + d * d * 4 + d * 4
            return offset + n_arms * per_arm

        for i in range(n_arms):
            arm = bandit.arms[i]

            arm.n = struct.unpack('<I', data[offset:offset + 4])[0]
            offset += 4

            a_size = d * d * 4
            arm.A = np.frombuffer(
                data[offset:offset + a_size], dtype=np.float32
            ).reshape(d, d).copy()
            offset += a_size

            arm.A_inv = np.frombuffer(
                data[offset:offset + a_size], dtype=np.float32
            ).reshape(d, d).copy()
            offset += a_size

            b_size = d * 4
            arm.b = np.frombuffer(
                data[offset:offset + b_size], dtype=np.float32
            ).copy()
            offset += b_size

        return offset

    # ── Thompson serialization ────────────────────────────────

    def _serialize_thompson(self, thompson: ThompsonSampler) -> bytes:
        """Serialize Thompson priors to bytes."""
        data = bytearray()
        data.extend(struct.pack('<I', thompson.n_arms))

        for alpha in thompson.alphas:
            data.extend(struct.pack('<d', alpha))
        for beta in thompson.betas:
            data.extend(struct.pack('<d', beta))

        return bytes(data)

    def _deserialize_thompson(
        self, data: bytes, offset: int, thompson: ThompsonSampler
    ) -> int:
        """Deserialize Thompson priors from bytes. Returns new offset."""
        n_arms = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4

        if n_arms != thompson.n_arms:
            logger.warning(
                f"Thompson arm count mismatch: file={n_arms}, "
                f"sampler={thompson.n_arms}"
            )
            return offset + n_arms * 16  # skip

        for i in range(n_arms):
            thompson.alphas[i] = struct.unpack(
                '<d', data[offset:offset + 8]
            )[0]
            offset += 8

        for i in range(n_arms):
            thompson.betas[i] = struct.unpack(
                '<d', data[offset:offset + 8]
            )[0]
            offset += 8

        return offset

    # ── Background thread ─────────────────────────────────────

    def start(
        self,
        sketch: CountMinSketch,
        bandit: LinUCBBandit,
        thompson: ThompsonSampler,
    ):
        """Start periodic background checkpoint saving."""
        self._running = True
        self._sketch = sketch
        self._bandit = bandit
        self._thompson = thompson

        self._thread = threading.Thread(
            target=self._save_loop,
            name="fsbs-checkpoint",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"Checkpoint thread started: "
            f"saving every {self.interval}s to {self.filepath}"
        )

    def _save_loop(self):
        """Background loop: save checkpoint every interval."""
        while self._running:
            # Sleep first — don't save immediately on startup
            # (give the system time to receive some data)
            for _ in range(int(self.interval)):
                if not self._running:
                    return
                time.sleep(1.0)

            if self._running:
                self.save(self._sketch, self._bandit, self._thompson)

    def stop(self):
        """Stop background thread and save final checkpoint."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        # Final save on shutdown
        if hasattr(self, '_sketch'):
            logger.info("Saving final checkpoint on shutdown...")
            self.save(self._sketch, self._bandit, self._thompson)

    def get_stats(self) -> dict:
        """Return checkpoint statistics."""
        return {
            'checkpoint_dir': self.checkpoint_dir,
            'checkpoint_file': self.filepath,
            'saves_completed': self.saves_completed,
            'saves_failed': self.saves_failed,
            'last_save_time': self.last_save_time,
            'last_save_size_bytes': self.last_save_size,
            'restore_success': self.restore_success,
            'restore_age_seconds': round(self.restore_age_seconds, 1),
            'interval_seconds': self.interval,
        }