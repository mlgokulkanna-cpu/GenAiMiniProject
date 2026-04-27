#!/bin/bash
# Start both backend and frontend concurrently

echo "🚀 ReviewAI — Starting all services..."
echo ""

# Kill background jobs on exit
trap 'kill $(jobs -p) 2>/dev/null' EXIT

# Start backend
echo "⚙️  Starting Backend (FastAPI) on port 8000..."
cd backend
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt -q
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting Frontend (React/Vite) on port 5173..."
cd frontend
npm install -q
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ All services running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for both
wait $BACKEND_PID $FRONTEND_PID
