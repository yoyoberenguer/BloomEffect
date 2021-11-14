# BUILD THE PROJECT WITH THE CORRECT PYTHON VERSION
# pip install wheel

# python setup.py bdist_wheel
# OR python setup.py sdist bdist_wheel (to include the source)

# for python 3.8
# C:\Users\yoann\AppData\Roaming\Python\Python38\Scripts\twine upload
# --verbose --repository testpypi dist/BloomEffect-1.0.3-cp38-cp38-win_amd64.whl

# for python 3.6
# C:\Users\yoann\AppData\Roaming\Python\Python36\Scripts\twine upload
# --verbose --repository testpypi dist/BloomEffect-1.0.3-cp36-cp36-win_amd64.whl

# python setup.py bdist_wheel
# twine upload --verbose --repository testpypi dist/*

# PRODUCTION v:
# version 1.0.2
# C:\Users\yoann\AppData\Roaming\Python\Python38\Scripts\twine upload --verbose dist/BloomEffect-1.0.2*

# CREATING EXECUTABLE
# pyinstaller --onefile pyinstaller_config.spec

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

import setuptools
from Cython.Build import cythonize
from setuptools import setup, Extension

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name                         ="BloomEffect",
    version                      ="1.0.1",
    author                       ="Yoann Berenguer",
    author_email                 ="yoyoberenguer@hotmail.com",
    description                  ="Pygame bloom effect (shader effect)",
    long_description             =long_description,
    long_description_content_type="text/markdown",
    url                          ="https://github.com/yoyoberenguer/BloomEffect",
    packages                     =setuptools.find_packages(),
    ext_modules                  =cythonize([
        Extension("bloom", ["bloom.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c")]),
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    license                      ='MIT',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Cython',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28'
    ],
    python_requires         ='>=3.0',
    platforms               =['any'],
    include_package_data    =True,
    data_files=[('./lib/site-packages/BloomEffect',
                 ['__init__.pxd',
                  '__init__.py',
                  'bloom.c',
                  'bloom.pxd',
                  'bloom.pyx',
                  'example.py',
                  'LICENSE',
                  'MANIFEST.in',
                  'pyproject.toml',
                  'README.md',
                  'requirements.txt',
                  'setup_bloom.py'


                  ]),
                ('./lib/site-packages/BloomEffect/tests',
                 [
                  'tests/__init__.py',
                  'tests/test_bloom.py',
                  'tests/profiling.py'
                 ]),

                ('./lib/site-packages/BloomEffect/Assets',
                 [
                  'Assets/Aliens.jpg',
                  'Assets/color_mask_circle.png',
                  'Assets/I1.png',
                  'Assets/i2.png',
                  'Assets/background_checker.png',
                  'Assets/bloom_bpf_values.png',
                  'Assets/bloom_smooth_values.png',
                  'Assets/control.png',
                  'Assets/i2_bloom.png',
                  'Assets/i3.png',
                  'Assets/i3_bloom.png',
                  'Assets/text_bloom.png'
                 ])
                ],

    project_urls = {  # Optional
                   'Bug Reports': 'https://github.com/yoyoberenguer/BloomEffect/issues',
                   'Source'     : 'https://github.com/yoyoberenguer/BloomEffect',
               },
)
