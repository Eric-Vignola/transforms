@echo off

python setup.py build_ext --inplace
move _array.pyd ..\_array.pyd