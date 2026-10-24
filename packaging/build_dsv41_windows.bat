@echo off
rem strata-dsv41.exe (the DeepSeek-V4.1 engine) for NVIDIA on Windows, built next to the release engine so package_engine.py ships it
rem beside strata.exe (setup.py dsv41_exe() looks for it in the engine folder).  cudart is static; the only extra DLL is cublas64_13.dll,
rem which package_engine.py copies from CUDA_PATH like it does for strata.exe.
rem   build_dsv41_windows.bat <build dir> [source dir]        env: STRATA_GGML_DIR (llama.cpp), CUDA_ROOT (default CUDA 13.0), BUILD_JOBS (default 4),
rem                                                            DSV41_ARCH (default "86-real;120" = sm_86 + sm_120 SASS, PTX sm_120)
rem Result: <build dir>\strata-dsv41.exe (about 66 MB).  Windows: only the cuda13 variant gets it (sm_120 needs CUDA 13).
setlocal
set "BD=%~1"
set "SRCD=%~2"
if "%BD%"=="" ( echo usage: build_dsv41_windows.bat ^<build dir^> [source dir] & exit /b 2 )
if "%SRCD%"=="" set "SRCD=%~dp0.."
if "%CUDA_ROOT%"=="" set "CUDA_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
if "%DSV41_ARCH%"=="" set "DSV41_ARCH=86-real;120"
if "%BUILD_JOBS%"=="" set "BUILD_JOBS=4"
if "%STRATA_GGML_DIR%"=="" ( echo set STRATA_GGML_DIR to the llama.cpp checkout & exit /b 2 )
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if not exist "%BD%\build.ninja" (
  cmake -G Ninja -S "%SRCD%" -B "%BD%" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=%DSV41_ARCH%" "-DCMAKE_CUDA_COMPILER=%CUDA_ROOT%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_ROOT%" "-DSTRATA_GGML_DIR=%STRATA_GGML_DIR%"
  if errorlevel 1 exit /b 1
)
start "" /belownormal /wait /b cmake --build "%BD%" -j %BUILD_JOBS% --target strata-dsv41
if errorlevel 1 exit /b 1
if not exist "%BD%\strata-dsv41.exe" ( echo no strata-dsv41.exe & exit /b 1 )
echo built %BD%\strata-dsv41.exe
exit /b 0
