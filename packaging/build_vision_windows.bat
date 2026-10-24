@echo off
rem The Windows CUDA picture reader for an engine package: strata-vision.exe built for the SAME architectures as the package's strata.exe.
rem
rem   packaging\build_vision_windows.bat cuda12 <stage folder>     CUDA 12.9: 60;61;70;75;80;86;89 (+ 89 PTX)  the Pascal / Volta package
rem   packaging\build_vision_windows.bat cuda13 <stage folder>     CUDA 13.0: 75;86;89;120 (+ 120 PTX)
rem
rem Why: the cuda12 and cuda13 packages used to share one strata-vision.exe, built before the Pascal code existed (unchanged since 0.1.44). On a
rem Tesla P100 / GTX 10 card the engine ran and the reader stopped with "IM2COL failed / device kernel image is invalid" (#3669), and the
rem app only said "the vision encoder did not start". Now each variant has its own reader and packaging\package_engine.py refuses a package
rem whose BUILD.json vision_archs do not cover its archs.
rem
rem Needs: Visual Studio 2022 Build Tools (run from a "x64 Native Tools" prompt, or let this call vcvars64), CMake + Ninja, the CUDA toolkit of
rem the variant (CUDA_PATH_V12_9 / CUDA_PATH_V13_0, or CUDA_PATH already pointing at it), and the pinned llama.cpp (python -c "import setup;
rem setup.get_llama_cpp()", or STRATA_GGML_DIR).
rem
rem Settings (all optional): STRATA_GGML_DIR  the llama.cpp checkout (default third_party\llama.cpp)   VISION_BUILD_DIR  the build folder
rem   VISION_ARCHS  override the architecture list (semicolons)      BUILD_JOBS  the number of compile jobs (default 4: the CUDA compiler uses RAM)
rem
rem It copies strata-vision.exe into <stage folder> and writes vision_archs into <stage folder>\BUILD.json. Nothing is uploaded.
setlocal EnableDelayedExpansion
set "VARIANT=%~1"
set "STAGE=%~2"
if "%VARIANT%"=="" goto usage
if "%STAGE%"=="" goto usage
for %%I in ("%~dp0..") do set "SRC=%%~fI"
if /i "%VARIANT%"=="cuda12" (
  set "ARCHS=60;61;70;75;80;86;89"
  if defined CUDA_PATH_V12_9 set "CUDA_PATH=%CUDA_PATH_V12_9%"
) else if /i "%VARIANT%"=="cuda13" (
  set "ARCHS=75;86;89;120"
  if defined CUDA_PATH_V13_0 set "CUDA_PATH=%CUDA_PATH_V13_0%"
) else goto usage
if defined VISION_ARCHS set "ARCHS=%VISION_ARCHS%"
if not defined CUDA_PATH (echo CUDA_PATH is not set: install the %VARIANT% toolkit & exit /b 1)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (echo no nvcc.exe in %CUDA_PATH%\bin & exit /b 1)
if not defined BUILD_JOBS set "BUILD_JOBS=4"
if not defined STRATA_GGML_DIR set "STRATA_GGML_DIR=%SRC%\third_party\llama.cpp"
set "LLAMA=%STRATA_GGML_DIR:\=/%"
if not exist "%LLAMA%/tools/mtmd" (echo llama.cpp with tools/mtmd not found in %LLAMA% - set STRATA_GGML_DIR & exit /b 1)
if not defined VISION_BUILD_DIR set "VISION_BUILD_DIR=%SRC%\build-vision-%VARIANT%-win"
if not exist "%STAGE%" mkdir "%STAGE%"

where cl >nul 2>nul || (
  for %%V in ("%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools" "%ProgramFiles%\Microsoft Visual Studio\2022\Community" "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools") do (
    if exist "%%~V\VC\Auxiliary\Build\vcvars64.bat" if not defined VCVARS set "VCVARS=%%~V\VC\Auxiliary\Build\vcvars64.bat"
  )
  if defined VCVARS call "!VCVARS!" >nul
)
where cl >nul 2>nul || (echo cl.exe not found: run this from an x64 Native Tools prompt & exit /b 1)

rem every arch real, and PTX for the newest one, the way the engine is built
set "REAL="
for %%A in (%ARCHS:;= %) do (
  set "LAST=%%A"
  if defined REAL (set "REAL=!REAL!;%%A-real") else (set "REAL=%%A-real")
)
cmake -G Ninja -S "%SRC%\tools\vision" -B "%VISION_BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="%LLAMA%" ^
  -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF ^
  "-DCMAKE_CUDA_ARCHITECTURES=!REAL!;!LAST!-virtual" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe" "-DCUDAToolkit_ROOT=%CUDA_PATH%" || exit /b 1
cmake --build "%VISION_BUILD_DIR%" --target strata-vision -j %BUILD_JOBS% || exit /b 1
copy /y "%VISION_BUILD_DIR%\bin\strata-vision.exe" "%STAGE%\strata-vision.exe" >nul || exit /b 1
py -3 "%~dp0package_engine.py" --stamp-vision "%STAGE%" --vision-archs "%ARCHS%" || exit /b 1
echo built: %STAGE%\strata-vision.exe  (%VARIANT%, archs %ARCHS%)
exit /b 0

:usage
echo usage: packaging\build_vision_windows.bat cuda12^|cuda13 ^<stage folder with strata.exe^>
exit /b 2
