"""
C++ 扩展构建脚本（pybind11 + setuptools）
用法: pip install .  或  python setup.py build_ext --inplace
依赖: pip install pybind11
"""
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 针对不同平台的编译参数
extra_compile_args = []

if sys.platform == "win32":
    # Windows/MSVC: 强制使用 UTF-8 编码，解决中文注释导致的 C4819 和 C2001 错误
    extra_compile_args = ["/utf-8"]
else:
    # Linux/MacOS: 开启优化
    extra_compile_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "cpp_hex_engine",
        ["hex_engine.cpp", "bindings.cpp"],
        include_dirs=["."],
        cxx_std=17,
        extra_compile_args=extra_compile_args,  # 关键修改：传入编译参数
    ),
]

setup(
    name="cpp_hex_engine",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)