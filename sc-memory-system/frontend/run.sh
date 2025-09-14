#!/bin/bash
# Install requirements
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.headless true