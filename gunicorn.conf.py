import os

# Gunicorn configuration file
# https://docs.gunicorn.org/en/stable/configure.html

# The socket to bind
bind = os.getenv("GENOS_API_BIND", "127.0.0.1:6001")

# Single worker to avoid loading the model into GPU memory multiple times
workers = 1

# Serve concurrent requests after the single model instance has loaded.
# Inference remains serialized by the application's inference lock.
worker_class = "gthread"
threads = max(2, int(os.getenv("GENOS_API_THREADS", "4")))

# The app module loads weights and completes warm-up before Gunicorn serves pages.
timeout = 300

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process name
proc_name = "genos_api"
