try:
    from setuptools import setup
    from setuptools import Extension
    from Cython.Build import cythonize
    import numpy
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


setup(
      name = 'my v2 cython',
      ext_modules = cythonize('inpaint_fmm.pyx'), 
      include_dirs=[numpy.get_include()]
)
