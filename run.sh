#!/bin/bash

# 백엔드 서버(app.py)를 백그라운드에서 실행
echo "Starting Flask server..."
python app.py &

# 5초 대기 (서버가 완전히 시작되도록)
sleep 5

# 프론트엔드 서버(webapp.py)를 실행
echo "Starting Streamlit app..."
streamlit run webapp.py
