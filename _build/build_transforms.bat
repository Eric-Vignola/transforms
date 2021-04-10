@echo off

python.exe _transforms.py
move _transforms.pyd ..\.
pause

