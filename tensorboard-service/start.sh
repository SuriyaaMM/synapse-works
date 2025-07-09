#!/bin/bash

# Install dependencies 
pip install --no-cache-dir -r requirements.txt

# Run TensorBoard on Render’s required public port 10000
tensorboard --logdir ../tbsummary --host 0.0.0.0 --port 10000
