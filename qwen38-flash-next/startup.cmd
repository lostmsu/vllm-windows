@echo off
rem Logon startup: serve on a network address with an API key and extra model
rem names, logging to startup.log. Put a shortcut to this file in the Startup
rem folder and set SERVE_HOST, SERVE_PORT and VLLM_API_KEY (read by vLLM) in the
rem user environment. Pass extra names with --served-model-name in SERVED_NAMES.
if "%GPU_UTIL%"=="" set GPU_UTIL=0.9
if "%SERVED_NAMES%"=="" set "SERVED_NAMES=nvidia/Qwen3.8-Flash-Next-FP4-Local qwen3.8-flash-next"
cd /d "%~dp0"
rem Right after logon the network may not have brought the address up yet.
if not "%SERVE_HOST%"=="" powershell -NoProfile -Command "for ($i = 0; $i -lt 120 -and -not (Get-NetIPAddress -IPAddress '%SERVE_HOST%' -ErrorAction SilentlyContinue); $i++) { Start-Sleep 5 }"
if exist startup.log move /y startup.log startup.prev.log >nul
call "%~dp0runenv.cmd" "%~dp0serve.cmd" --served-model-name %SERVED_NAMES% > startup.log 2>&1
