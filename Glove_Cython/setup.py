from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

# setup(
#     ext_modules = cythonize("glove_cython.pyx")
# )


# setup(ext_modules = cythonize('spacy_ex.pyx', compiler_directives={'language' : "c++"}),
# 	include_dirs = [numpy.get_include()])


extensions = [
    Extension("glove_cython", sources=["glove_cython.pyx"], include_dirs=[numpy.get_include()], extra_compile_args=["-O3"])
]

setup(
    name="glove_cython",
    ext_modules = cythonize(extensions),
)