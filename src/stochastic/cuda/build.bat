@echo off
REM CUDA FGN Kernel Build Script for Windows
REM Requires: CUDA Toolkit with nvcc, cuFFT, and cuRAND
REM           Visual Studio with C++ build tools

setlocal

REM Set temp directory to avoid path issues with special characters
set TEMP=C:\Temp
set TMP=C:\Temp
if not exist C:\Temp mkdir C:\Temp

REM Initialize Visual Studio environment (adjust path if needed)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d "%~dp0"

echo Building CUDA FGN kernel...

REM GPU architecture - adjust for your GPU
REM Common values: sm_75 (Turing), sm_80 (Ampere), sm_86 (Ampere), sm_89 (Ada Lovelace)
set ARCH=sm_89

echo Target architecture: %ARCH%

nvcc -O3 -use_fast_math -arch=%ARCH% ^
    -shared fgn.cu ^
    -o ./fgn_windows/fgn.dll ^
    -lcufft -lcurand

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo Built: fgn_windows/fgn.dll
echo Done!
