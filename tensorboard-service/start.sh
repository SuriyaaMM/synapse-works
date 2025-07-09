#!/bin/bash

# Install Python dependencies (if needed)
pip install --no-cache-dir tensorboard

# Start TensorBoard on Render's public port (10000)
tensorboard --logdir ../tbsummary --host 0.0.0.0 --port 10000
