@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 1251 >nul
pushd "%~dp0"

python "%~dp0runner.py" %*
exit /b %errorlevel%
