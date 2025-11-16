#!/bin/bash
# Start the Streamlit UI

echo "Starting Streamlit UI on http://localhost:8501..."
export API_BASE_URL="http://localhost:8001"
poetry run streamlit run ui/app.py --server.port 8501
