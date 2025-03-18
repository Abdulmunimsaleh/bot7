#!/bin/bash
# Install Playwright browsers with minimal dependencies
python -m playwright install chromium --with-deps=false

# Set memory limit options for Python process
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export PYTHONHASHSEED=0

# Run with limited memory and aggressive garbage collection
PYTHONPATH=$PYTHONPATH:. uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --limit-concurrency 4 --backlog 8