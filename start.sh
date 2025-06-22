#!/bin/bash

# Start backend (Node.js)
npm start & 
echo "started backend"

# Start Python worker
python service/worker.py & 
echo "started python backend"

# Start frontend
cd frontend && npm run dev & 
echo "local session started at: http://localhost:5173/"

# Wait for all background processes (optional)
wait
