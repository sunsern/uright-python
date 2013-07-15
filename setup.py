from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import uright

ext_modules = [
    Extension("uright._dtwc", 
              ["uright/_dtwc.pyx"],
              libraries=["m"])]

setup(name='uRight',
      version=uright.__version__,
      author='Sunsern Cheamanunkul',
      author_email='sunsern@gmail.com',
      packages=['uright',],
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
