#!/bin/bash
# Start the ReviewAI backend
set -e

echo "🚀 Starting ReviewAI Backend..."

# Check for .env
if [ ! -f .env ]; then
  echo "⚠️  .env file not found. Copying from .env.example..."
  cp .env.example .env 2>/dev/null || echo "Please create a .env file with your API keys."
fi

# Create venv if not exists
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

echo "✅ Starting FastAPI server on http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
