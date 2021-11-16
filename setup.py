"""
Setup.py file

Configure the project, build the package and upload the package to PYPI
"""
import setuptools
from Cython.Build import cythonize
from setuptools import Extension

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
                      "\nTry: \n   C:\\pip install numpy on a window command prompt.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BloomEffect",
    version="1.0.2",
    author="Yoann Berenguer",
    author_email="yoyoberenguer@hotmail.com",
    description="Pygame bloom effect (shader effect)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyoberenguer/BloomEffect",
    # packages=setuptools.find_packages(),
    packages=['BloomEffect'],
    # ext_package='BloomEffect',
    ext_modules=cythonize([
        Extension("BloomEffect.bloom", ["BloomEffect/bloom.pyx"],
                  extra_compile_args=["/openmp", "/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c")]),
    include_dirs=[numpy.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    license='MIT',

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
        'Cython>=0.28',
        'numpy>=1.18',
        'pygame>=2.0'
    ],
    python_requires='>=3.6',
    platforms=['any'],
    include_package_data=True,
    data_files=[
        ('./lib/site-packages/BloomEffect',
         ['LICENSE',
          'MANIFEST.in',
          'pyproject.toml',
          'README.md',
          'requirements.txt',
          'BloomEffect/__init__.py',
          'BloomEffect/__init__.pxd',
          'BloomEffect/bloom.pyx',
          'BloomEffect/bloom.pxd',
          'BloomEffect/deprecated.pyx',
          'BloomEffect/setup_bloom.py',
          'BloomEffect/example.py'
          ]),
        ('./lib/site-packages/BloomEffect/tests',
         ['BloomEffect/tests/test_bloom.py',
          'BloomEffect/tests/profiling.py'
          ]),
        ('./lib/site-packages/BloomEffect/Assets',
         [
             'BloomEffect/Assets/Aliens.jpg',
             'BloomEffect/Assets/color_mask_circle.png',
             'BloomEffect/Assets/I1.png',
             'BloomEffect/Assets/i2.png',
             'BloomEffect/Assets/background_checker.png',
             'BloomEffect/Assets/bloom_bpf_values.png',
             'BloomEffect/Assets/bloom_smooth_values.png',
             'BloomEffect/Assets/control.png',
             'BloomEffect/Assets/i2_bloom.png',
             'BloomEffect/Assets/i3.png',
             'BloomEffect/Assets/i3_bloom.png',
             'BloomEffect/Assets/text_bloom.png'
         ])
    ],

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/yoyoberenguer/BloomEffect/issues',
        'Source': 'https://github.com/yoyoberenguer/BloomEffect',
    },
)

