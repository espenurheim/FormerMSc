from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'former_class_cpp',  # Module name
        sources=['bindings.cpp', 'former_class.cpp'],
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-fvisibility=default']  # Compiler flags
    ),
]

setup(
    name='former_class_cpp',
    version='0.0.1',
    ext_modules=ext_modules,
)