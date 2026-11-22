@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

rem Prefer the official Windows Python launcher.
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        py -3 main.py
        goto after_run
    )
)

rem Then try a real python command.
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 (
        python main.py
        goto after_run
    )
)

rem Look in common per-user Python and Conda locations.
if exist "%USERPROFILE%\anaconda3\python.exe" (
    "%USERPROFILE%\anaconda3\python.exe" main.py
    goto after_run
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    "%USERPROFILE%\miniconda3\python.exe" main.py
    goto after_run
)
if exist "C:\ProgramData\anaconda3\python.exe" (
    "C:\ProgramData\anaconda3\python.exe" main.py
    goto after_run
)
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%~fD\python.exe" (
        "%%~fD\python.exe" main.py
        goto after_run
    )
)

echo.
echo [错误] 未找到可用的 Python 3.10+ 解释器。
echo 请安装 Python 3.12 或 Anaconda，也可以在 Spyder 中直接打开并运行 main.py。
echo.
pause
exit /b 1

:after_run
set "RUN_STATUS=%ERRORLEVEL%"
if not "%RUN_STATUS%"=="0" (
    echo.
    echo [错误] 程序异常退出（退出码 %RUN_STATUS%）。
    echo 如果是首次运行提示缺少模块，请先双击运行 "安装依赖.bat"。
    pause
)
exit /b %RUN_STATUS%
