#!/bin/bash

# EURON ML Automation - Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

echo "🚀 Starting EURON ML Automation..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt --quiet

# Create necessary directories
mkdir -p data models reports

# Start FastAPI backend in background
echo -e "${GREEN}Starting FastAPI backend on http://localhost:8000${NC}"
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Streamlit frontend
echo -e "${GREEN}Starting Streamlit frontend on http://localhost:8501${NC}"
cd frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=================================="
echo -e "${GREEN}🎉 EURON ML Automation is running!${NC}"
echo ""
echo "📊 Frontend (Streamlit): http://localhost:8501"
echo "🔧 Backend (FastAPI):    http://localhost:8000"
echo "📚 API Docs:             http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo "=================================="

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait
