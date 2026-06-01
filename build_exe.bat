@echo off
setlocal
cd /d %~dp0
pyinstaller -F -w -n FourierLab main.py
echo.
echo Build finished. Check the dist folder.
pause
