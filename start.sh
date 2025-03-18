#!/bin/bash
# Install Playwright browsers
python -m playwright install chromium --force

# Set memory limit options for Python process
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536

# Start the FastAPI app with memory monitoring
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
