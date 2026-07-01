@echo off
chcp 65001 >nul
title CloudKnight Live Engine

echo ============================================
echo   CloudKnight - 实时交易引擎 (Live)
echo ============================================
echo.
echo 正在检查依赖...
call python -c "import akquant, akshare" >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 缺少依赖，正在安装...
    call pip install -r requirements.txt -q
)

echo 启动实时交易引擎...
echo 按 Ctrl+C 停止
echo.
call python main.py live

pause
