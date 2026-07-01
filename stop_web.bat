@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   CloudKnight - 停止所有服务
echo ============================================
echo.

set KILLED=0

REM 终止 8080 端口的 web 服务
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING" 2^>nul') do (
    echo 停止 Web 仪表盘 (PID=%%a)...
    taskkill /PID %%a /F >nul 2>&1
    set /a KILLED+=1
)

REM 终止运行 main.py 的 python 进程
for /f "tokens=2 delims=," %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH 2^>nul') do (
    set PIDVAR=%%~a
    if defined PIDVAR (
        echo 停止 Python 进程 !PIDVAR!...
        taskkill /PID !PIDVAR! /F >nul 2>&1
        set /a KILLED+=1
    )
)

echo.
if !KILLED! gtr 0 (
    echo 已停止 !KILLED! 个进程。
) else (
    echo 没有发现运行中的服务。
)
echo.
echo 按任意键退出...
pause >nul
