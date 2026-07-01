@echo off
chcp 65001 >nul
title CloudKnight - 前后端启动器
cd /d %~dp0

echo ============================================
echo   CloudKnight - 前后端并行启动
echo ============================================
echo.

REM 清理残留进程
echo [0/4] 清理 8080 端口残留进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING" 2^>nul') do (
    echo   发现残留进程 PID=%%a，正在终止...
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM 清理缓存
echo [1/4] 清理 Python 缓存...
call python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]" 2>nul

REM 检查依赖
echo [2/4] 检查依赖...
call python -c "import akquant, akshare, fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo   安装缺失依赖...
    call pip install -r requirements.txt -q
)
echo   依赖 OK
echo.

REM 启动实时交易引擎（后端）
echo [3/4] 启动实时交易引擎（后端）...
start "CloudKnight Live Engine" cmd /k "cd /d %~dp0 && title CloudKnight Live Engine && python main.py live"
timeout /t 2 /nobreak >nul

REM 启动 Web 仪表盘（前端）
echo [4/4] 启动 Web 仪表盘（前端）...
start "CloudKnight Web Dashboard" cmd /k "cd /d %~dp0 && title CloudKnight Web Dashboard && python main.py web"

echo.
echo ============================================
echo   已启动两个独立窗口:
echo     - 实时交易引擎 (后端)
echo     - 数据仪表盘    (http://localhost:8080)
echo ============================================
echo.
echo   关闭本窗口不影响已启动的服务
echo   用 stop_web.bat 可一键停止所有服务

pause
