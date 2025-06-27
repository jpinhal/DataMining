@echo off

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" streamlit_env

streamlit run app.py
pause