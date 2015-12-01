REM configuration of paths
set VSFORPYTHON="C:\Program Files (x86)\Common Files\Microsoft\Visual C++ for Python\9.0"

REM add tdm gcc stuff
set PATH=C:\TDM-GCC-32\bin;C:\TDM-GCC-32\mingw32;%PATH%

REM add winpython stuff
CALL "C:\WinPython-64bit-2.7.10.3\scripts\env.bat"

REM configure path for msvc compilers
REM CALL %VSFORPYTHON%\vcvarsall.bat
CALL "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64\vcvarsx86_amd64.bat" amd64

REM return a shell
cmd.exe /k