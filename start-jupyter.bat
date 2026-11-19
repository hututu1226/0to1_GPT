@echo off
rem 双击这个文件就能打开 JupyterLab
cd /d "%~dp0"

if not exist ".venv\Scripts\jupyter-lab.exe" (
    echo [X] 没找到 .venv\Scripts\jupyter-lab.exe
    echo     请先在仓库根目录建好环境：
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r env\requirements.txt jupyterlab ipykernel
    pause
    exit /b 1
)

echo 启动 JupyterLab ... 浏览器会自动打开，关掉这个窗口就停服务
.venv\Scripts\jupyter-lab.exe --notebook-dir "%cd%" --ServerApp.token="" --ServerApp.password=""
pause
