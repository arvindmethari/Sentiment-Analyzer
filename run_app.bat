@echo off
title Sentiment Analyzer AI
echo ===================================================
echo Starting Sentiment Analyzer AI (Modal Dark Mode)
echo ===================================================
cd /d "G:\JYESTA\Projects\Minor\Sentiment_Analyzer"
call .venv\Scripts\activate.bat
streamlit run app.py --server.port=8501
pause
