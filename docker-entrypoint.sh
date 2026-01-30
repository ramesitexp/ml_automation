#!/bin/bash

# Start FastAPI backend
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 3

# Start Streamlit frontend
cd /app/frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Keep container running
wait
