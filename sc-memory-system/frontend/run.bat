@echo off
REM Install requirements
pip install -r requirements.txt

REM Run Streamlit app
streamlit run streamlit_app.py --server.port 8501 --server.headless true