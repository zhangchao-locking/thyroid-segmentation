@echo off
echo 启动甲状腺结节分割系统...
echo.
cd /d "%~dp0"
python -m streamlit run app.py
pause
