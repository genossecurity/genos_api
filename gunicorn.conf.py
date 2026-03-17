import os

# Gunicorn configuration file
# https://docs.gunicorn.org/en/stable/configure.html

# The socket to bind
bind = os.getenv("GENOS_API_BIND", "127.0.0.1:6001")

# Single worker to avoid loading the model into GPU memory multiple times
workers = 1

# The type of workers to use
worker_class = "sync"

# Timeout for workers — must be long enough to cover model loading + inference.
# The worker loads both CodeBERT models and runs a warm-up pass on startup;
# once loaded, individual requests are fast (<1s).
timeout = 300

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process name
proc_name = "genos_api"
