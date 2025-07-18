#!/bin/bash
# Setup script for Zoom Meeting Transcript API

echo "🔧 Setting up Zoom Meeting Transcript API..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "🔄 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "🔄 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 To start the server:"
echo "   ./start_server.sh"
echo ""
echo "🧪 To test the API:"
echo "   source venv/bin/activate"
echo "   python test_api_comprehensive.py"
echo ""
echo "📖 API documentation will be available at:"
echo "   http://localhost:8000/docs"
