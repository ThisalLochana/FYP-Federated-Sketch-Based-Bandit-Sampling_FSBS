"""
Feature Extractor — converts a raw OTLP span into a 4-byte feature vector.

Architecture reference:
  - latency_bucket: 3 bits [0-7] — which exponential latency range
  - has_error:      1 bit  [0-1] — any 5xx or exception
  - svc_cluster_id: 8 bits [0-255] — pre-assigned service ID
  - topo_hash_prefix: 16 bits — rolling hash of service call path

Total: 28 bits packed into 4 bytes (32-bit integer)
Extraction cost: ~50 nanoseconds (3 integer comparisons + 1 hash)
"""

from typing import Optional, Dict, Any
import hashlib


# Latency bucket boundaries (microseconds)
# Exponential ranges as described in the architecture
LATENCY_BUCKETS_US = [
    1_000,       # bucket 0: 0–1ms
    5_000,       # bucket 1: 1–5ms
    20_000,      # bucket 2: 5–20ms
    50_000,      # bucket 3: 20–50ms
    200_000,     # bucket 4: 50–200ms
    500_000,     # bucket 5: 200–500ms
    2_000_000,   # bucket 6: 500ms–2s
                 # bucket 7: >2s
]

# Service name → cluster ID mapping
# Pre-assigned integer IDs for each service in the Google microservices demo
# These fit in 8 bits (0–255)
SERVICE_CLUSTER_IDS: Dict[str, int] = {
    "frontend":                0,
    "productcatalogservice":   1,
    "cartservice":             2,
    "currencyservice":         3,
    "checkoutservice":         4,
    "paymentservice":          5,
    "shippingservice":         6,
    "emailservice":            7,
    "recommendationservice":   8,
    "adservice":               9,
    "loadgenerator":           10,
    # Reserve IDs 11–255 for future services
}

# Default for unknown services
DEFAULT_CLUSTER_ID = 255


class FeatureVector:
    """
    The 4-byte feature vector that describes a span's "character".
    
    Bit layout (MSB to LSB):
    [31..16] topo_hash_prefix (16 bits)
    [15..8]  svc_cluster_id   (8 bits)  
    [7]      has_error         (1 bit)
    [6..4]   latency_bucket    (3 bits)
    [3..0]   reserved / novelty_bucket (4 bits, added later)
    """
    
    __slots__ = [
        'latency_bucket', 'has_error', 'svc_cluster_id',
        'topo_hash_prefix', 'novelty_score', 'packed_key', 'arm_index'
    ]
    
    def __init__(
        self,
        latency_bucket: int,
        has_error: int,
        svc_cluster_id: int,
        topo_hash_prefix: int,
        novelty_score: float = 1.0,
    ):
        self.latency_bucket = latency_bucket & 0x7         # 3 bits
        self.has_error = has_error & 0x1                     # 1 bit
        self.svc_cluster_id = svc_cluster_id & 0xFF         # 8 bits
        self.topo_hash_prefix = topo_hash_prefix & 0xFFFF   # 16 bits
        self.novelty_score = novelty_score                    # float [0,1]
        
        # Pack into 32-bit key for sketch lookup
        self.packed_key = (
            (self.topo_hash_prefix << 16) |
            (self.svc_cluster_id << 8) |
            (self.has_error << 7) |
            (self.latency_bucket << 4)
        )
        
        # Arm index for the bandit (8-bit svc_cluster + top 4 bits of topo)
        # = 256 possible arms maximum
        self.arm_index = (
            (self.svc_cluster_id << 4) | (self.topo_hash_prefix >> 12)
        ) & 0xFF  # clamp to 256 arms
    
    def to_bandit_context(self) -> list:
        """
        Convert to the 4-element float vector used by LinUCB.
        
        This is the 'x' vector in the UCB formula:
            UCB = x^T(A^{-1}b) + α * sqrt(x^T A^{-1} x)
        
        We normalize each feature to [0, 1] range for numerical stability.
        """
        return [
            self.latency_bucket / 7.0,        # [0, 1]
            float(self.has_error),              # 0 or 1
            self.svc_cluster_id / 255.0,        # [0, 1]
            self.novelty_score,                 # [0, 1] — from sketch
        ]
    
    def __repr__(self):
        return (
            f"FeatureVector(lat_bucket={self.latency_bucket}, "
            f"error={self.has_error}, svc={self.svc_cluster_id}, "
            f"topo=0x{self.topo_hash_prefix:04X}, "
            f"novelty={self.novelty_score:.3f}, "
            f"arm={self.arm_index})"
        )


class FeatureExtractor:
    """
    Extracts a FeatureVector from an OTLP span.
    
    This is the first component on the hot path.
    It reads span metadata and computes the 4-byte feature vector.
    No I/O, no allocation, just integer comparisons and one hash.
    """
    
    def __init__(self, service_name: str):
        """
        Args:
            service_name: The name of THIS service (the one the sidecar
                          is attached to). Used for svc_cluster_id.
        """
        self.service_name = service_name
        self.svc_cluster_id = SERVICE_CLUSTER_IDS.get(
            service_name, DEFAULT_CLUSTER_ID
        )
    
    def compute_latency_bucket(self, duration_us: int) -> int:
        """
        Map span duration to one of 8 exponential buckets.
        
        Args:
            duration_us: Span duration in microseconds
        
        Returns:
            Bucket index 0–7
        """
        for i, boundary in enumerate(LATENCY_BUCKETS_US):
            if duration_us < boundary:
                return i
        return 7  # >2 seconds
    
    def compute_topo_hash(self, parent_services: list) -> int:
        """
        Compute a 16-bit topology hash from the ordered list of services
        seen so far in this trace's path.
        
        This captures the "shape" of the trace — e.g., 
        [frontend, checkout, payment] has a different hash than
        [frontend, productcatalog].
        
        In production, this would be a rolling hash maintained in the
        trace context header. Here we compute it from the parent service list.
        
        Args:
            parent_services: List of service names in call order
                             e.g., ["frontend", "checkoutservice"]
        
        Returns:
            Top 16 bits of a 32-bit hash
        """
        if not parent_services:
            return 0
        
        # Build a string from the ordered service IDs
        path_str = "|".join(
            str(SERVICE_CLUSTER_IDS.get(s, DEFAULT_CLUSTER_ID))
            for s in parent_services
        )
        # Use a fast hash, take top 16 bits
        h = hashlib.md5(path_str.encode(), usedforsecurity=False).digest()
        return (h[0] << 8) | h[1]  # first 2 bytes = 16 bits
    
    def extract(
        self,
        span_data: Dict[str, Any],
    ) -> FeatureVector:
        """
        Extract features from a span.
        
        Args:
            span_data: Dictionary with span fields:
                - 'service_name': str — originating service
                - 'duration_us': int — span duration in microseconds
                - 'status_code': int — 0=OK, 1=CANCELLED, 2=ERROR
                - 'parent_services': list[str] — ordered services in trace path
                - 'attributes': dict — span attributes/tags
        
        Returns:
            FeatureVector ready for sketch query and bandit decision
        """
        # 1. Latency bucket (3 bits)
        duration_us = span_data.get('duration_us', 0)
        latency_bucket = self.compute_latency_bucket(duration_us)
        
        # 2. Error flag (1 bit)
        status_code = span_data.get('status_code', 0)
        attributes = span_data.get('attributes', {})
        has_error = 1 if (
            status_code == 2 or  # OTLP StatusCode ERROR
            attributes.get('error', False) or
            attributes.get('http.status_code', 200) >= 500
        ) else 0
        
        # 3. Service cluster ID (8 bits)
        svc_name = span_data.get('service_name', self.service_name)
        svc_cluster_id = SERVICE_CLUSTER_IDS.get(svc_name, DEFAULT_CLUSTER_ID)
        
        # 4. Topology hash prefix (16 bits)
        parent_services = span_data.get('parent_services', [])
        topo_hash = self.compute_topo_hash(parent_services)
        
        return FeatureVector(
            latency_bucket=latency_bucket,
            has_error=has_error,
            svc_cluster_id=svc_cluster_id,
            topo_hash_prefix=topo_hash,
        )
    
    def extract_from_otlp_span(
        self,
        span,
        resource_attributes: Dict[str, str],
    ) -> FeatureVector:
        """
        Extract features directly from a protobuf OTLP Span object.
        
        This is the method called in production when receiving
        actual OTLP data from the microservice.
        
        Args:
            span: opentelemetry.proto.trace.v1.Span protobuf object
            resource_attributes: dict of resource-level attributes
                                 (contains service.name)
        
        Returns:
            FeatureVector
        """
        # Duration: span has start_time_unix_nano and end_time_unix_nano
        duration_ns = span.end_time_unix_nano - span.start_time_unix_nano
        duration_us = duration_ns // 1000  # convert to microseconds
        
        # Service name from resource attributes
        svc_name = resource_attributes.get('service.name', self.service_name)
        
        # Status code
        status_code = span.status.code if span.status else 0
        
        # Attributes — convert from protobuf KeyValue list to dict
        attributes = {}
        for kv in span.attributes:
            key = kv.key
            val = kv.value
            if val.HasField('string_value'):
                attributes[key] = val.string_value
            elif val.HasField('int_value'):
                attributes[key] = val.int_value
            elif val.HasField('bool_value'):
                attributes[key] = val.bool_value
        
        # Build span_data dict and use the standard extract method
        span_data = {
            'service_name': svc_name,
            'duration_us': duration_us,
            'status_code': status_code,
            'parent_services': [],  # will be populated from trace context
            'attributes': attributes,
        }
        
        return self.extract(span_data)