@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
call start_windows.bat --install
set "status=%errorlevel%"
popd
exit /b %status%
