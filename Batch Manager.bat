@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 1251 >nul
set "CALLER_DIR=%cd%"
pushd "%~dp0"

if "%~1"=="" (
  if exist "%CALLER_DIR%\full-batch.plan" (
    python "%~dp0utils\batch-manager.py" "%CALLER_DIR%\full-batch.plan"
  ) else (
    python "%~dp0utils\batch-manager.py" "%CALLER_DIR%"
  )
) else (
  python "%~dp0utils\batch-manager.py" %*
)
exit /b %errorlevel%
