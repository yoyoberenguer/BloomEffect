from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

ext_modules = cythonize(Extension(
    'bloom', ['bloom.pyx'],
    extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"],
    language="c",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    )
)

setup(
  name="PygameEffect",
  cmdclass={"build_ext": build_ext},
  ext_modules=ext_modules,
  include_dirs=[numpy.get_include()]
)