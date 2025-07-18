#!/bin/bash
# Startup script for Zoom Meeting Transcript API

echo "🚀 Starting Zoom Meeting Transcript API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Run: python3 -m venv venv"
    echo "   Then: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment and start server
echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "🔄 Starting FastAPI server..."
echo "📍 Server will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python main.py
