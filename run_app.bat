@echo off
echo Starting Multi-layer Grad-CAM Video Inspector...
echo.
echo The app will be available at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run app.py --server.port 8501
pause