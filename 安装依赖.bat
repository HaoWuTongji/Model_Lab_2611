@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
)
if "%PY_CMD%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
        if not errorlevel 1 set "PY_CMD=python"
    )
)
if "%PY_CMD%"=="" if exist "%USERPROFILE%\anaconda3\python.exe" set "PY_CMD=\"%USERPROFILE%\anaconda3\python.exe\""
if "%PY_CMD%"=="" if exist "%USERPROFILE%\miniconda3\python.exe" set "PY_CMD=\"%USERPROFILE%\miniconda3\python.exe\""
if "%PY_CMD%"=="" if exist "C:\ProgramData\anaconda3\python.exe" set "PY_CMD=\"C:\ProgramData\anaconda3\python.exe\""
if "%PY_CMD%"=="" (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
        if exist "%%~fD\python.exe" set "PY_CMD=\"%%~fD\python.exe\""
    )
)

if "%PY_CMD%"=="" (
    echo [错误] 未找到可用的 Python 3 解释器，请先安装 Python 或 Anaconda。
    pause
    exit /b 1
)

echo 正在使用 %PY_CMD% 安装 ModelLab 所需依赖...
%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [提示] 如果下载较慢或失败，可尝试使用国内镜像：
    echo   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

echo.
echo 依赖安装完成！现在可以双击 start_windows.bat 启动 ModelLab。
pause
