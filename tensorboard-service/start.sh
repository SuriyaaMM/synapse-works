#!/bin/bash

# Start TensorBoard on Render's only available port: 10000
tensorboard --logdir ../tbsummary --host 0.0.0.0 --port 10000
