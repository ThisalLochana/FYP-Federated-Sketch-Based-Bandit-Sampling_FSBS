"""
HTTP API for the FSBS Sidecar.

Endpoints:
  GET  /health              → {"status": "ok"}
  GET  /metrics             → full metrics JSON
  GET  /arms                → list of active arms with stats
  GET  /decisions?limit=N   → recent decisions
  POST /reward              → receive reward signal

Runs on port 8081 alongside the gRPC server (port 4317).
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger('fsbs-http')

# Module-level references set by start_http_server()
_sampler = None
_trace_service = None


class FSBSHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the sidecar API."""

    def log_message(self, format, *args):
        """Suppress default access logs — too noisy."""
        pass

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        """Read JSON from request body."""
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8'))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')
        params = parse_qs(parsed.query)

        if path == '/health':
            self._handle_health()
        elif path == '/metrics':
            self._handle_metrics()
        elif path == '/arms':
            self._handle_arms()
        elif path == '/decisions':
            limit = int(params.get('limit', ['50'])[0])
            self._handle_decisions(limit)
        else:
            self._send_json({'error': 'not found', 'path': self.path}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        if path == '/reward':
            self._handle_reward()
        else:
            self._send_json({'error': 'not found'}, 404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    # ---- Endpoint handlers ----

    def _handle_health(self):
        self._send_json({'status': 'ok'})

    def _handle_metrics(self):
        data = {}
        if _sampler:
            data['sampler'] = _sampler.get_metrics()
        if _trace_service:
            data['service'] = _trace_service.get_stats()
        self._send_json(data)

    def _handle_arms(self):
        if not _sampler:
            self._send_json({'arms': []})
            return
        arms = _sampler.get_active_arms()
        self._send_json({
            'active_arms': len(arms),
            'confident_arms': sum(1 for a in arms if a['confident']),
            'arms': arms[:50],  # top 50 by observation count
        })

    def _handle_decisions(self, limit: int):
        if not _sampler:
            self._send_json({'decisions': []})
            return
        decisions = _sampler.get_recent_decisions(limit=min(limit, 200))
        self._send_json({
            'count': len(decisions),
            'decisions': decisions,
        })

    def _handle_reward(self):
        """
        Receive a reward signal.

        Expected body:
        {
            "arm_index": 5,
            "context": [0.5, 0.0, 0.1, 0.8],
            "reward": 0.8
        }
        """
        if not _sampler:
            self._send_json({'error': 'sampler not ready'}, 503)
            return

        try:
            body = self._read_json()
            arm_index = int(body['arm_index'])
            context = [float(v) for v in body['context']]
            reward = float(body['reward'])

            if not (0.0 <= reward <= 1.0):
                self._send_json({'error': 'reward must be 0.0-1.0'}, 400)
                return
            if len(context) != 4:
                self._send_json({'error': 'context must have 4 elements'}, 400)
                return

            result = _sampler.process_reward(arm_index, context, reward)
            self._send_json({
                'status': 'ok',
                **result,
            })

        except (KeyError, ValueError, TypeError) as e:
            self._send_json(
                {'error': f'invalid payload: {e}'}, 400
            )
        except Exception as e:
            logger.error(f"Reward processing error: {e}")
            self._send_json({'error': str(e)}, 500)


def start_http_server(sampler, trace_service, port: int = 8081):
    """
    Start the HTTP API server in a background thread.

    Args:
        sampler: FSBSSampler instance
        trace_service: FSBSTraceService instance
        port: HTTP port to listen on
    """
    global _sampler, _trace_service
    _sampler = sampler
    _trace_service = trace_service

    server = HTTPServer(('0.0.0.0', port), FSBSHandler)

    thread = threading.Thread(
        target=server.serve_forever,
        name="fsbs-http-api",
        daemon=True,
    )
    thread.start()
    logger.info(f"HTTP API listening on 0.0.0.0:{port}")
    return server