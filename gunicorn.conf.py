import multiprocessing
import os

# Gunicorn configuration file
# https://docs.gunicorn.org/en/stable/configure.html

# The socket to bind
bind = "127.0.0.1:5000"

# Single worker to avoid loading the model into GPU memory multiple times
workers = 1

# The type of workers to use
worker_class = "sync"

# Timeout for workers
timeout = 120

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process name
proc_name = "genos_api"
