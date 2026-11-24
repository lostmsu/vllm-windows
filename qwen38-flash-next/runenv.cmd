@echo off
rem Runtime environment for vLLM on Windows: MSVC (FlashInfer JIT), CUDA_HOME, venv.
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_PATH=%CUDA_HOME%"
for %%I in ("%~dp0..") do set "VIRTUAL_ENV=%%~fI\.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%CUDA_HOME%\bin;%PATH%"
%*
