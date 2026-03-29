"""
FSBS Sidecar — main entry point.

This is the OTLP-speaking gRPC server that:
  1. Receives spans from microservices (port 4317)
  2. Runs them through the FSBS decision pipeline
  3. Forwards SAMPLED spans to the real OTel Collector / Jaeger
  4. Drops non-sampled spans

It replaces the OTel Collector as the first hop for trace data.

Data flow:
  Microservice ──OTLP──▶ THIS SIDECAR ──OTLP──▶ OTel Collector ──▶ Jaeger
                          (decides SAMPLE/DROP)
"""

import os
import sys
import time
import signal
import logging
import json
import threading
from concurrent import futures

import grpc
from google.protobuf import json_format

# OpenTelemetry protobuf types
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.proto.trace.v1 import trace_pb2
from opentelemetry.proto.common.v1 import common_pb2

# gRPC client for forwarding
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

# Our FSBS components
from fsbs.sampler import FSBSSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger('fsbs-sidecar')


class FSBSTraceService(trace_service_pb2_grpc.TraceServiceServicer):
    """
    gRPC service that implements the OTLP TraceService.
    
    Microservices think they're talking to a normal OTel Collector.
    Instead, they're talking to our FSBS sidecar, which makes
    intelligent sampling decisions before forwarding.
    """
    
    def __init__(
        self,
        sampler: FSBSSampler,
        forward_endpoint: str,
        service_name: str,
    ):
        """
        Args:
            sampler: The FSBS sampler instance
            forward_endpoint: Where to send sampled spans (e.g., "otel-collector:4317")
            service_name: Name of the service this sidecar is attached to
        """
        self.sampler = sampler
        self.forward_endpoint = forward_endpoint
        self.service_name = service_name
        
        # gRPC channel for forwarding sampled spans
        self._forward_channel = grpc.insecure_channel(forward_endpoint)
        self._forward_stub = trace_service_pb2_grpc.TraceServiceStub(
            self._forward_channel
        )
        
        logger.info(f"FSBS TraceService forwarding to {forward_endpoint}")
    
    def Export(self, request, context):
        """
        Handle incoming ExportTraceServiceRequest.
        
        This is the gRPC method that microservices call to send spans.
        
        Flow:
          1. For each span in the request, run FSBS decision
          2. Collect sampled spans into a new request
          3. Forward the filtered request to the real collector
        """
        sampled_resource_spans = []
        
        for resource_spans in request.resource_spans:
            # Extract resource attributes (contains service.name)
            resource_attrs = {}
            if resource_spans.resource:
                for kv in resource_spans.resource.attributes:
                    if kv.value.HasField('string_value'):
                        resource_attrs[kv.key] = kv.value.string_value
                    elif kv.value.HasField('int_value'):
                        resource_attrs[kv.key] = str(kv.value.int_value)
            
            svc_name = resource_attrs.get('service.name', self.service_name)
            
            sampled_scope_spans = []
            
            for scope_spans in resource_spans.scope_spans:
                sampled_spans = []
                
                for span in scope_spans.spans:
                    # Build span_data dict for the sampler
                    span_data = self._span_to_dict(span, svc_name)
                    
                    # Run FSBS decision
                    decision = self.sampler.decide(span_data)
                    
                    if decision.should_sample:
                        sampled_spans.append(span)
                        logger.debug(
                            f"SAMPLE: {svc_name}/{span.name} "
                            f"[{decision.method}] score={decision.score:.3f}"
                        )
                    else:
                        logger.debug(
                            f"DROP:   {svc_name}/{span.name} "
                            f"[{decision.method}] score={decision.score:.3f}"
                        )
                
                if sampled_spans:
                    # Build a new ScopeSpans with only sampled spans
                    new_scope_spans = trace_pb2.ScopeSpans()
                    if scope_spans.scope:
                        new_scope_spans.scope.CopyFrom(scope_spans.scope)
                    new_scope_spans.spans.extend(sampled_spans)
                    sampled_scope_spans.append(new_scope_spans)
            
            if sampled_scope_spans:
                # Build a new ResourceSpans with only sampled scope spans
                new_resource_spans = trace_pb2.ResourceSpans()
                if resource_spans.resource:
                    new_resource_spans.resource.CopyFrom(resource_spans.resource)
                new_resource_spans.scope_spans.extend(sampled_scope_spans)
                sampled_resource_spans.append(new_resource_spans)
        
        # Forward sampled spans to the real collector
        if sampled_resource_spans:
            forward_request = trace_service_pb2.ExportTraceServiceRequest()
            forward_request.resource_spans.extend(sampled_resource_spans)
            
            try:
                self._forward_stub.Export(forward_request)
            except grpc.RpcError as e:
                logger.error(f"Failed to forward spans: {e}")
        
        # Return success to the microservice
        return trace_service_pb2.ExportTraceServiceResponse()
    
    def _span_to_dict(self, span, service_name: str) -> dict:
        """
        Convert a protobuf Span to the dict format expected by
        FeatureExtractor.extract().
        """
        # Duration in microseconds
        duration_ns = span.end_time_unix_nano - span.start_time_unix_nano
        duration_us = max(duration_ns // 1000, 0)
        
        # Status code (0=UNSET, 1=OK, 2=ERROR)
        status_code = span.status.code if span.HasField('status') else 0
        
        # Span attributes
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
            elif val.HasField('double_value'):
                attributes[key] = val.double_value
        
        # Trace ID as hex string
        trace_id = span.trace_id.hex() if span.trace_id else ''
        
        return {
            'trace_id': trace_id,
            'span_id': span.span_id.hex() if span.span_id else '',
            'service_name': service_name,
            'operation_name': span.name,
            'duration_us': duration_us,
            'status_code': status_code,
            'attributes': attributes,
            'parent_services': [],  # populated from trace context in Phase 3+
        }


class MetricsReporter:
    """
    Periodically logs FSBS sampler metrics to console.
    This makes it easy to see how the sampler is performing.
    """
    
    def __init__(self, sampler: FSBSSampler, interval: float = 10.0):
        self.sampler = sampler
        self.interval = interval
        self._running = True
        self._thread = threading.Thread(
            target=self._report_loop,
            name="fsbs-metrics",
            daemon=True,
        )
        self._thread.start()
    
    def _report_loop(self):
        while self._running:
            time.sleep(self.interval)
            metrics = self.sampler.get_metrics()
            logger.info(
                f"FSBS Metrics | "
                f"total={metrics['total_spans']} "
                f"sampled={metrics['sampled_spans']} "
                f"dropped={metrics['dropped_spans']} "
                f"rate={metrics['sample_rate']:.1%} | "
                f"thompson={metrics['thompson_decisions']} "
                f"linucb={metrics['linucb_decisions']} "
                f"forced={metrics['forced_samples']} | "
                f"queue={metrics['queue_size']} "
                f"q_dropped={metrics['queue_dropped']}"
            )
    
    def stop(self):
        self._running = False


def serve():
    """Start the FSBS sidecar gRPC server."""
    
    # Configuration from environment variables
    service_name = os.environ.get('FSBS_SERVICE_NAME', 'unknown')
    listen_port = os.environ.get('FSBS_LISTEN_PORT', '4317')
    forward_endpoint = os.environ.get('FSBS_FORWARD_ENDPOINT', 'otel-collector:4317')
    alpha = float(os.environ.get('FSBS_ALPHA', '1.0'))
    threshold = float(os.environ.get('FSBS_THRESHOLD', '0.5'))
    confidence_threshold = int(os.environ.get('FSBS_CONFIDENCE_THRESHOLD', '10'))
    metrics_interval = float(os.environ.get('FSBS_METRICS_INTERVAL', '10'))
    
    logger.info("=" * 60)
    logger.info("FSBS Sidecar starting")
    logger.info(f"  Service:          {service_name}")
    logger.info(f"  Listen:           0.0.0.0:{listen_port}")
    logger.info(f"  Forward to:       {forward_endpoint}")
    logger.info(f"  Alpha:            {alpha}")
    logger.info(f"  Threshold:        {threshold}")
    logger.info(f"  Confidence:       {confidence_threshold}")
    logger.info("=" * 60)
    
    # Create the sampler
    sampler = FSBSSampler(
        service_name=service_name,
        alpha=alpha,
        threshold=threshold,
        confidence_threshold=confidence_threshold,
        force_sample_errors=True,
    )
    
    # Start metrics reporter
    metrics_reporter = MetricsReporter(sampler, interval=metrics_interval)
    
    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=[
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),  # 16MB
        ],
    )
    
    # Register the FSBS trace service
    trace_service = FSBSTraceService(
        sampler=sampler,
        forward_endpoint=forward_endpoint,
        service_name=service_name,
    )
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(
        trace_service, server
    )
    
    # Start listening
    server.add_insecure_port(f'0.0.0.0:{listen_port}')
    server.start()
    logger.info(f"FSBS Sidecar listening on 0.0.0.0:{listen_port}")
    
    # Handle shutdown signals
    shutdown_event = threading.Event()
    
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()
    
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    
    # Wait for shutdown
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    logger.info("Shutting down FSBS Sidecar...")
    metrics_reporter.stop()
    sampler.shutdown()
    server.stop(grace=5)
    
    # Print final metrics
    final_metrics = sampler.get_metrics()
    logger.info(f"Final metrics: {json.dumps(final_metrics, indent=2)}")
    logger.info("FSBS Sidecar stopped")


if __name__ == '__main__':
    serve()