'''
python setup.py build_ext -i
to compile
'''

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Enforce Python 3
if sys.version_info.major < 3:
    raise RuntimeError("This package requires Python 3.x")

# Define the extension
extension = Extension(
    "mesh_core_cython",  # Output module name
    sources=["mesh_core_cython.pyx", "mesh_core.cpp"],  # Cython and C++ sources
    language="c++",  # Specify C++ language
    include_dirs=[numpy.get_include()],  # NumPy headers
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_2_0_API_VERSION')]  # NumPy 2.0 API
)

setup(
    name="mesh_core_cython",
    ext_modules=cythonize(
        [extension],
        compiler_directives={'language_level': '3str'}  # Python 3 string semantics
    ),
    python_requires=">=3.6",  # Minimum Python 3 version
)
