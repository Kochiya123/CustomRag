# Gunicorn configuration file for production WSGI server
import multiprocessing
import os

# Server socket
# Use PORT environment variable provided by Render, fallback to 8000 for local development
port = int(os.getenv('PORT', 8000))
bind = f"0.0.0.0:{port}"
backlog = 2048

# Worker processes
# IMPORTANT: CUDA cannot be re-initialized in forked processes
# Use 1 worker for CUDA/GPU workloads to avoid "Cannot re-initialize CUDA" error
# If you need more workers, each must initialize CUDA separately (preload_app=False)
workers = int(os.getenv('GUNICORN_WORKERS', 1))  # Default to 1 for CUDA compatibility
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 5 minutes for long-running ML inference
keepalive = 5

# Logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'rag-api'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (uncomment and configure if using SSL)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Performance tuning
max_requests = 1000  # Restart workers after this many requests to prevent memory leaks
max_requests_jitter = 50  # Add randomness to prevent all workers restarting at once
# IMPORTANT: preload_app must be False for CUDA compatibility
# CUDA contexts cannot be shared across forked processes
preload_app = False  # Disable preload - each worker initializes CUDA separately

# Graceful timeout for worker shutdown
graceful_timeout = 30

