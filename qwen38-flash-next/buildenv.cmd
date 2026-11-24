@echo off
rem Build environment for vllm-windows: MSVC x64 + CUDA 13.4 + Python 3.14 venv, sm_120 only.
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
rem Pin the VS 2026 compiler (a VS 2022 cl.exe is also on the base PATH).
set "CC=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
set "CXX=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
set "CUDAHOSTCXX=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
set VLLM_VERSION_OVERRIDE=0.34.0+cu134
set CMAKE_POLICY_VERSION_MINIMUM=3.5
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_ROOT=%CUDA_HOME%"
for %%I in ("%~dp0..") do set "REPO=%%~fI"
set "PATH=%CUDA_HOME%\bin;%REPO%\.venv\Scripts;%PATH%"
set "VIRTUAL_ENV=%REPO%\.venv"
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=16
set NVCC_THREADS=1
set CMAKE_BUILD_TYPE=Release
rem Rust frontend TLS (openssl-sys): link the installed OpenSSL statically (/MD libs).
set "OPENSSL_DIR=C:\Program Files\OpenSSL-Win64"
set "OPENSSL_INCLUDE_DIR=C:\Program Files\OpenSSL-Win64\include"
set "OPENSSL_LIB_DIR=C:\Program Files\OpenSSL-Win64\lib\VC\x64\MD"
set OPENSSL_STATIC=1
set OPENSSL_LIBS=libssl_static:libcrypto_static
rem Long paths for git clones made by CMake FetchContent (env-only, no config change).
set GIT_CONFIG_COUNT=1
set GIT_CONFIG_KEY_0=core.longpaths
set GIT_CONFIG_VALUE_0=true
cd /d "%REPO%"
%*
