@echo off
echo ========================================
echo   甲状腺结节分割系统 - 环境安装
echo ========================================
echo.

echo [1/4] 检查 Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 未安装，正在打开下载页面...
    echo 请下载 Python 3.8+ 并安装
    start https://www.python.org/downloads/
    pause
    exit
)
echo ✅ Python 已安装

echo.
echo [2/4] 安装依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ 依赖安装失败
    pause
    exit
)
echo ✅ 依赖安装完成

echo.
echo [3/4] 检查模型文件...
if not exist "results\nnUNet\best_model.pth" (
    echo ⚠️ 警告: 未找到 nnUNet 模型文件
)
if not exist "results\MKUNet\best_model.pth" (
    echo ⚠️ 警告: 未找到 MKUNet 模型文件
)

echo.
echo [4/4] 启动应用...
echo.
echo ========================================
echo   环境安装完成！
echo   正在启动分割系统...
echo ========================================
echo.
python -m streamlit run app.py

pause
