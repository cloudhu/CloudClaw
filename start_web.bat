@echo off
chcp 65001 >nul
title CloudKnight Web Dashboard
cd /d %~dp0

echo ============================================
echo   CloudKnight - 数据仪表盘 (Web)
echo ============================================
echo.

REM 清理 8080 端口残留进程
echo [0/3] 检查 8080 端口占用...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING" 2^>nul') do (
    echo   发现残留进程 PID=%%a，正在终止...
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM 清理 Python 缓存
echo [1/3] 清理 Python 缓存...
call python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]; print('   OK')" 2>nul

REM 检查依赖
echo [2/3] 检查依赖...
call python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo   安装缺失依赖...
    call pip install fastapi uvicorn -q
)
echo   依赖 OK

echo [3/3] 启动仪表盘: http://127.0.0.1:8080
echo   按 Ctrl+C 停止服务
echo.
call python main.py web

pause
