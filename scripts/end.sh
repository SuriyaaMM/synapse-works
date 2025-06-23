#!/bin/bash

echo "Killing Node backend..."
pkill -f "npm start"

echo "Killing Python backend..."
pkill -f "python service/worker.py"

echo "Killing Frontend dev server..."
pkill -f "vite"

echo "All services stopped."
