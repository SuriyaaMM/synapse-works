# start.sh
mkdir -p logs

# Start Redis
redis-server > logs/redis.log 2>&1 &
echo "started redis (logs/redis.log)"

# Wait for Redis to be ready
until redis-cli ping > /dev/null 2>&1; do
  echo "waiting for redis..."
  sleep 0.5
done

# Start Node backend
npm start > logs/backend.log 2>&1 &
echo "started backend (logs/backend.log)"

# Start Python worker
python service/worker.py > logs/python_worker.log 2>&1 & 
echo "started python backend (logs/python_worker.log)"

# Start frontend
cd frontend && npm run dev > ../logs/frontend.log 2>&1 &
echo "local session started at: http://localhost:5173/ (logs/frontend.log)"

wait
