@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
set "project_python=.venv\Scripts\python.exe"
set "install="
if /i "%~1"=="--install" set "install=1"
if exist "%project_python%" goto project_environment

rem A fresh install prefers Python 3.12. On an existing installation, find an
rem interpreter that already has the application dependencies before starting.
set "interpreter="
set "selector="
set "probe=import sys; sys.exit(sys.version_info < (3, 10))"
if not defined install set "probe=import sys; from importlib.util import find_spec; sys.exit(sys.version_info < (3, 10) or any(find_spec(p) is None for p in ('uvicorn','fastapi','pydantic','multipart','httpx','PIL','highspy','dotenv','openpyxl')))"
call :try_python py -3.12
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
call :try_python python
call :try_python py -3.11
call :try_python py -3.10
call :try_python py -3
call :try_python "%USERPROFILE%\anaconda3\python.exe"
call :try_python "%USERPROFILE%\miniconda3\python.exe"
call :try_python "%ProgramData%\anaconda3\python.exe"
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if not defined interpreter call :try_python "%%~fD\python.exe"
if not defined interpreter goto missing_python
if not defined install goto launch

echo 正在创建项目虚拟环境 .venv ...
"%interpreter%" %selector% -m venv .venv
if errorlevel 1 goto failed

:project_environment
set "interpreter=%project_python%"
set "selector="
if not defined install goto launch
echo 正在为项目虚拟环境安装依赖 ...
"%interpreter%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo 安装完成，可以双击 start_windows.bat 启动 ModelLab。
pause
popd
exit /b 0

:launch
"%interpreter%" %selector% main.py
if errorlevel 1 goto failed
popd
exit /b 0

:try_python
if defined interpreter exit /b 0
"%~1" %~2 -c "%probe%" >nul 2>&1
if errorlevel 1 exit /b 1
set "interpreter=%~1"
set "selector=%~2"
exit /b 0

:missing_python
if defined install echo [错误] 找不到 Python 3.10 或以上版本。请先安装 Python 3.12，然后重新运行安装依赖.bat。
if not defined install echo [错误] 找不到已安装 ModelLab 依赖的环境。请先双击安装依赖.bat。
set "status=1"
goto show_error

:failed
set "status=%errorlevel%"
echo [错误] 命令执行失败，退出码 %status%。请查看上方报错。
if not defined install echo 如果缺少依赖，请运行安装依赖.bat。

:show_error
pause
popd
exit /b %status%
