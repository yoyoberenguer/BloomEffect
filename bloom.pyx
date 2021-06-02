###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

"""
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DEFINITION:
Bloom effet is a computer graphics effect used in video games, demos,
and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.
Requirements:
- Pygame 3
- Numpy
- Cython (C extension for python)
- A C compiler for windows (Visual Studio, MinGW etc) install on your system
  and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is install on your system,
  refer to external documentation or tutorial in order to setup this process.
  e.g https://devblogs.microsoft.com/python/unable-to-find-vcvarsall-bat/


METHOD:
Acronyme : bpf (Bright Pass Filter)

1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
  bpf24_b_c or bpf32_b_c (adjust the luminence threshold value to get the best filter effect).

2) Downscale the newly created bpf image into sub-surface factor x2, x4, x8, x16 using
   pygame transform.scale method. No need to use smoothscale (bilinear filtering method).

3) Apply a Gaussian blur 5x5 filter on each of the downsized bpf images (if variable smooth_ is > 1,
   then the Gaussian filter 5x5 will by applied more than once. Note, this will have little effect
   on the final image quality.

4) Re-scale to orinial sizes all the bpf images using a bilinear filtering method.
   Note : Using an un-filtered rescaling method will pixelate the final output image.
   Recommandation: For best performances sets smoothscale acceleration.
   A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
   'SSE' allows SSE extensions as well.

5) Blit all the bpf images on the original pygame surface (input image), use pygame additive
   blend mode effect.

BUILDING THE PROJECT:
In a command prompt and under the directory containing the source files
C:\>python setup_bloom.py build_ext --inplace
If the compilation fail, refers to the requirement section and make sure cython
and a C-compiler are correctly install on your system.

TWO METHODS:
# Below bloom method is using massively numpy.ndarray to manipulate data.
bloom = bloom_effect_array(surface, threshold, smooth_=1)

# Below method is using BufferProxy or C-Buffer data structure.
# It is also the fastest algorithm available.
bloom = bloom_effect_buffer(surface, threshold, smooth_=1)

Surface   : pygame.Surface to be bloom (compatible 24, 32-bit format)
threshold : Integer value for the bright pass filter, filter threshold value
smooth_   : Smooth define the quantity of Gaussian blur5x5 kernel passes that will be
            applied to all sub-surface (default is 1, vertical & horizontal)
            Note the Gaussian algorithm is cpu demanding. 4 is plenty smoothing
Left image with smooth_=1, right image with smooth_=10

TIPS:
python test_bloom.py
If you get the following error message after execution, refer to
the section building the project.
<<ModuleNotFoundError: No module named 'bloom'>>


DEMO VERSION:
Decompress the archive in the source directory
In order to work, demo.exe must be under the same directory with i2.jpg
Copy/move i2.jpg if necessary to demo.exe location.


TIMING:
print(timeit.timeit("bloom_effect_array(im, 255, smooth_=1)",
                    "from __main__ import bloom_effect_array, im", number=10) / 10)
print(timeit.timeit("bloom_effect_buffer(im, 255, smooth_=1)",
                     "from __main__ import bloom_effect_buffer, im", number=10) / 10)
Method 1:
bloom_effect_array
texture 600x600 24-bit gives a modest 0.0793 (79ms) processing time

Method 2:
bloom_effect_buffer
texture 600x600 24-bit gives a modest 0.04516 (45ms) processing time

Those values are not too bad considering that all the texture processing is done
entirely by the CPU.
Soon I will implement a mask method that will improve efficieny of both techniques.

LINKS:
https://learnopengl.com/Advanced-Lighting/Bloom
https://kalogirou.net/2006/05/20/how-to-do-good-bloom-for-hdr-rendering/
https://catlikecoding.com/unity/tutorials/advanced-rendering/bloom/

"""

__version__ = 1.00

try:
    import pygame
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")
    raise SystemExit

try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")
    raise SystemExit


try:
    import numpy
    from numpy import asarray, uint8, float32, zeros, float64
except ImportError:
    print("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")



# MAPPING LIBRARY IS REQUIRED
try:
    import MAPPING
except ImportError:
    raise ImportError("\n<MAPPING> library is missing on your system.")

try:
    from MAPPING cimport xyz, to1d_c, to3d_c, vfb_rgb_c, vfb_c, vmap_buffer_c
except ImportError:
    raise ImportError("\n<MAPPING> Cannot import methods.")

cimport numpy as np
from libc.math cimport sin, sqrt, cos, atan2, pi, round, floor, fmax, fmin, pi, tan, exp, ceil, fmod
from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX, qsort, malloc, free, abs
import timeit

# TODO
# 1)Create a method equivalent of pygame.Surface.copy() very slow



DEF THREADS = 8
DEF METHOD = 'static'

# --------------------------- INTERFACE ----------------------------------

cpdef blur5x5_buffer24(rgb_buffer, width, height, depth, mask=None):
    return blur5x5_buffer24_c(rgb_buffer, width, height, depth, mask=None)

cpdef blur5x5_buffer32(rgba_buffer, width, height, depth, mask=None):
    return blur5x5_buffer32_c(rgba_buffer, width, height, depth, mask=None)

cpdef blur5x5_array24(rgb_array_, mask=None):
    return blur5x5_array24_c(rgb_array_, mask=None)

cpdef blur5x5_array32(rgb_array_, mask=None):
    return blur5x5_array32_c(rgb_array_, mask=None)

# ******* METHODS THAT CAN BE ACCESS DIRECTLY FROM PYTHON SCRIPT ********

cpdef bloom_effect_buffer24(surface_, threshold_, smooth_, mask_=None):
    return bloom_effect_buffer24_c(surface_, threshold_, smooth_, mask_=None)

cpdef bloom_effect_buffer32(surface_, int threshold_, int smooth_, mask_=None):
    return bloom_effect_buffer32_c(surface_, threshold_, smooth_, mask_=None)

cpdef bloom_effect_array24(surface_, threshold_, smooth_, mask_=None):
    return bloom_effect_array24_c(surface_, threshold_, smooth_, mask_=None)

cpdef bloom_effect_array32(surface_, threshold_, smooth_, mask_=None):
    return bloom_effect_array32_c(surface_, threshold_, smooth_, mask_=None)

cpdef scale_array24_mult(rgb_array):
    return scale_array24_mult_c(rgb_array)

# --------------------------- IMPLEMENTATION -----------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blur5x5_buffer24_c(unsigned char [::1] rgb_buffer,
                      int width, int height, int depth, mask=None):
    """
    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values

    :param depth: integer; image depth (3)RGB, default 3
    :param height: integer; image height
    :param width:  integer; image width
    :param rgb_buffer: 1d buffer representing a 24bit format pygame.Surface  
    :return: 24-bit Pygame.Surface without per-pixel information and array 
    """

    cdef:
        int b_length = len(rgb_buffer)


    # check if the buffer length equal theoretical length
    if b_length != (width * height * depth):
        print("\nIncorrect 24-bit format image "
              "expecting %s bytes got %s " % (b_length, width * height * depth))

    # kernel 5x5 separable
    cdef:
        float [::1] kernel = \
            numpy.array(([1.0/16.0,
                          4.0/16.0,
                          6.0/16.0,
                          4.0/16.0,
                          1.0/16.0]), dtype=numpy.float32, copy=False)

        short int kernel_half = 2
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [::1] convolve = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [::1] convolved = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [:] buffer_ = rgb_buffer

    with nogil:
        # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array (width, height, depth)
            # v.y point to the column value ...
            # v.z is always = 0 as the i value point always
            # to the red color of a pixel in the C-buffer structure
            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 3)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                # v.x correspond to the row index value in a 3d array
                # v.x is always pointing to the red color of a pixel (see for i loop with
                # step = 3) in the C-buffer data structure.
                xx = v.x + kernel_offset

                # avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red = buffer_[index]
                    green = buffer_[index + 1]
                    blue = buffer_[index + 2]

                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

                index = vmap_buffer_c(i, width, height, 3)

                r, g, b = 0, 0, 0

                v = to3d_c(index, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[index    ] = <unsigned char>r
                convolved[index + 1] = <unsigned char>g
                convolved[index + 2] = <unsigned char>b

    return pygame.image.frombuffer(convolve, (width, height), "RGB"), convolve


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef blur5x5_buffer32_c(unsigned char [:] rgba_buffer,
                      int width, int height, int depth, mask=None):
    """
    Method using a C-buffer as input image (width * height * depth) uint8 data type
    5 x5 Gaussian kernel used:
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    It uses convolution property to process the image in two passes (horizontal and vertical passes).
    Pixels convoluted outside the image edges will be set to an adjacent pixel edge values
   
    :param depth: integer; image depth (3)RGB, default 3
    :param height: integer; image height
    :param width:  integer; image width
    :param rgba_buffer: 1d buffer representing a 24bit format pygame.Surface  
    :return: 24-bit Pygame.Surface without per-pixel information and array 
    """

    cdef:
        int b_length= len(rgba_buffer)


    # check if the buffer length equal theoretical length
    if b_length != (width * height * depth):
        raise ValueError(
            "\nIncorrect 32-bit format image, "
            "expecting %s got %s " % (width * height * depth, b_length))

    # kernel 5x5 separable
    cdef:
        float [::1] kernel = \
            numpy.array(([1.0/16.0,
                          4.0/16.0,
                          6.0/16.0,
                          4.0/16.0,
                          1.0/16.0]), dtype=numpy.float32, copy=False)

        short int kernel_half = 2
        int xx, yy, index, i, ii
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue
        xyz v;

        # convolve array contains pixels of the first pass(horizontal convolution)
        # convolved array contains pixels of the second pass.
        # buffer_ source pixels
        unsigned char [:] convolve = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [:] convolved = numpy.empty(width * height * depth, numpy.uint8)
        unsigned char [:] buffer_ = numpy.frombuffer(rgba_buffer, numpy.uint8)

    with nogil:
        # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array (width, height, depth)
            # v.y point to the column value ...
            # v.z is always = 0 as the i value point always
            # to the red color of a pixel in the C-buffer structure
            v = to3d_c(i, width, depth)

            # testing
            # index = to1d_c(v.x, v.y, v.z, width, 4)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                # v.x correspond to the row index value in a 3d array
                # v.x is always pointing to the red color of a pixel (see for i loop with
                # step = 4) in the C-buffer data structure.
                xx = v.x + kernel_offset

                # avoid buffer overflow
                if xx < 0 or xx > (width - 1):
                    red, green, blue = 0, 0, 0

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d_c(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red   = buffer_[index    ]
                    green = buffer_[index + 1]
                    blue  = buffer_[index + 2]

                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i    ] = <unsigned char>r
            convolve[i + 1] = <unsigned char>g
            convolve[i + 2] = <unsigned char>b
            convolve[i + 3] = buffer_[i + 3]

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule=METHOD, num_threads=THREADS):

                index = vmap_buffer_c(i, width, height, depth)

                r, g, b = 0, 0, 0

                v = to3d_c(index, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0 or yy > (height-1):

                        red, green, blue = 0, 0, 0
                    else:

                        ii = to1d_c(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[index    ] = <unsigned char>r
                convolved[index + 1] = <unsigned char>g
                convolved[index + 2] = <unsigned char>b
                convolved[index + 3] = buffer_[index + 3]

    return pygame.image.frombuffer(convolved, (width, height), "RGBA"), convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array24_c(unsigned char [:, :, :] rgb_array_, mask=None):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array_: numpy.ndarray type (w, h, 3) uint8 
    :return: Return 24-bit a numpy.ndarray type (w, h, 3) uint8
    """


    cdef int w, h, dim
    try:
        w, h, dim = (<object>rgb_array_).shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    # kernel_ = numpy.array(([1.0 / 16.0,
    #                        4.0 / 16.0,
    #                        6.0 / 16.0,
    #                        4.0 / 16.0,
    #                        1.0 / 16.0]), dtype=float32, copy=False)

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = kernel_
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 3), dtype=uint8)
        short int kernel_length = len(kernel)
        int x, y, xx, yy
        float k, r, g, b, s
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=METHOD, num_threads=THREADS):  # range [0..h-1)

            for x in range(0, w):  # range [0..w-1]

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=METHOD, num_threads=THREADS):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1], convolved[x, y, 2] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b

    return convolved




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] blur5x5_array32_c(unsigned char [:, :, :] rgb_array_, mask=None):
    """
    # Gaussian kernel 5x5
        # |1   4   6   4  1|
        # |4  16  24  16  4|
        # |6  24  36  24  6|  x 1/256
        # |4  16  24  16  4|
        # |1  4    6   4  1|
    This method is using convolution property and process the image in two passes,
    first the horizontal convolution and last the vertical convolution
    pixels convoluted outside image edges will be set to adjacent edge value
    
    :param rgb_array_: 3d numpy.ndarray type (w, h, 4) uint8, RGBA values
    :return: Return a numpy.ndarray type (w, h, 4) uint8
    """

    cdef int w, h, dim
    try:
        w, h, dim = rgb_array_.shape[:3]

    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    # kernel_ = numpy.array(([1.0 / 16.0,
    #                        4.0 / 16.0,
    #                        6.0 / 16.0,
    #                        4.0 / 16.0,
    #                        1.0 / 16.0]), dtype=float32, copy=False)

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = kernel_
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
        unsigned char [:, :, ::1] convolved = numpy.empty((w, h, 4), dtype=uint8)
        short int kernel_length = len(kernel)
        int x, y, xx, yy
        float k, r, g, b
        char kernel_offset
        unsigned char red, green, blue

    with nogil:
        # horizontal convolution
        for y in prange(0, h, schedule=METHOD, num_threads=THREADS):

            for x in range(0, w):

                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    xx = x + kernel_offset

                    # check boundaries.
                    # Fetch the edge pixel for the convolution
                    if xx < 0:
                        red, green, blue = rgb_array_[0, y, 0],\
                        rgb_array_[0, y, 1], rgb_array_[0, y, 2]
                    elif xx > (w - 1):
                        red, green, blue = rgb_array_[w-1, y, 0],\
                        rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
                    else:
                        red, green, blue = rgb_array_[xx, y, 0],\
                            rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
                    <unsigned char>g, <unsigned char>b

        # Vertical convolution
        for x in prange(0,  w, schedule=METHOD, num_threads=THREADS):

            for y in range(0, h):
                r, g, b = 0, 0, 0

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]
                    yy = y + kernel_offset

                    if yy < 0:
                        red, green, blue = convolve[x, 0, 0],\
                        convolve[x, 0, 1], convolve[x, 0, 2]
                    elif yy > (h -1):
                        red, green, blue = convolve[x, h-1, 0],\
                        convolve[x, h-1, 1], convolve[x, h-1, 2]
                    else:
                        red, green, blue = convolve[x, yy, 0],\
                            convolve[x, yy, 1], convolve[x, yy, 2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[x, y, 0], convolved[x, y, 1],\
                convolved[x, y, 2], convolved[x, y, 3] = \
                    <unsigned char>r, <unsigned char>g, <unsigned char>b, rgb_array_[x, y, 3]

    return convolved



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf24_c(image, int threshold = 128, bint transpose=False):
    """
    Bright pass filter compatible 24-bit 
    
    Bright pass filter for 24bit image (method using 3d array data structure)
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param transpose: Transpose the final array (width and height are transpose if True)
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a Pygame Surface and a 3d numpy.ndarray format (w, h, 3) 
    (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image)

    # make sure the surface is 24-bit format RGB
    if not image.get_bitsize() == 24:
        raise ValueError('Surface is not 24-bit format.')

    try:
        rgb_array = pygame.surfarray.pixels3d(image)
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int w, h
    w, h = rgb_array.shape[:2]

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)


    cdef:
        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, :, ::1] out_rgb= numpy.empty((w, h, 3), numpy.uint8)
        unsigned char [:, :, ::1] out_rgb_transposed = numpy.empty((h, w, 3), numpy.uint8)
        int i = 0, j = 0
        float lum, c

    if transpose is not None and transpose==True:
        with nogil:
            for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
                for j in range(h):
                    # ITU-R BT.601 luma coefficients
                    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114

                    if lum > threshold:
                        c = (lum - threshold) / lum
                        out_rgb_transposed[j, i, 0] = <unsigned char>(rgb[i, j, 0] * c)
                        out_rgb_transposed[j, i, 1] = <unsigned char>(rgb[i, j, 1] * c)
                        out_rgb_transposed[j, i, 2] = <unsigned char>(rgb[i, j, 2] * c)
                    else:
                        out_rgb_transposed[j, i, 0] = 0
                        out_rgb_transposed[j, i, 1] = 0
                        out_rgb_transposed[j, i, 2] = 0

        return pygame.image.frombuffer(out_rgb_transposed, (w, h), 'RGB'), out_rgb_transposed
    else:
        with nogil:
            for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
                for j in range(0, h):
                    # ITU-R BT.601 luma coefficients
                    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
                    if lum > threshold:
                        c = (lum - threshold) / lum
                        out_rgb[i, j, 0] = <unsigned char>(rgb[i, j, 0] * c)
                        out_rgb[i, j, 1] = <unsigned char>(rgb[i, j, 1] * c)
                        out_rgb[i, j, 2] = <unsigned char>(rgb[i, j, 2] * c)
                    else:
                        out_rgb[i, j, 0], out_rgb[i, j, 1], out_rgb[i, j, 2] = 0, 0, 0

        return pygame.image.frombuffer(out_rgb, (w, h), 'RGB'), out_rgb



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf24_b_c(image, int threshold = 128, bint transpose=False):
    """
    Bright pass filter for 24bit image (method using c-buffer)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param transpose:boolean; True | False , transpose the final array / image. 
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a 24 bit pygame.Surface filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    cdef:
        int w = image.get_width(), h = image.get_height()
        unsigned short int mode = 0;

    # make sure the surface is 24-bit format RGB
    if not image.get_bitsize() == 24:
        raise ValueError('Surface is not 24-bit format.')

    try:
        # BGR BUFFER
        # THIS IS THE FASTEST WAY TO GET THE BUFFER.
        # IF THE BUFFER IS NOT CONTIGUOUS, THIS WILL THROW
        # AN ERROR MESSAGE.
        #
        # buffer_ = numpy.asarray(im.get_view('2')).copy('C')
        # buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8)

        buffer_ = image.get_view('2')
        buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        mode = 1
    except:
        try:
            # RGB BUFFER
            # SLOWEST METHOD BUT WORKS EVEN IF THE BUFFER IS NOT
            # CONTIGUOUS
            buffer_ = pygame.image.tostring(image, 'RGB')
            buffer_ = numpy.frombuffer(buffer_, dtype=numpy.uint8).copy()   #.copy('C')
            mode = 0
        except:
            raise ValueError('\nInvalid surface.')

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        # int b_length = buffer_.length
        int b_length = len(buffer_)
        unsigned char [:] c_buffer = buffer_
        unsigned char [:] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0, index =0, tmp
        float lum, c

    if transpose:
        # FINAL BUFFER IS TRANSPOSE USING METHOD vmap_buffer_c
        # IF ARRAY IS NOT SYMMETRIC ROWS AND COLUMNS ARE SWAPPED
        if w != h:
            tmp = w
            w = h
            h = tmp
        with nogil:
            for i in prange(0, b_length, 3, schedule=METHOD, num_threads=THREADS):
                # ITU-R BT.601 luma coefficients
                lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114

                index = vmap_buffer_c(i, w, h, depth=3)

                if lum > threshold:
                    c = (lum - threshold) / lum
                    if mode == 0:
                        # RGB
                        out_buffer[index    ] = <unsigned char>(c_buffer[i    ] * c)
                        out_buffer[index + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[index + 2] = <unsigned char>(c_buffer[i + 2] * c)
                    else:
                        # BGR
                        out_buffer[index    ] = <unsigned char>(c_buffer[i + 2] * c)
                        out_buffer[index + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[index + 2] = <unsigned char>(c_buffer[i    ] * c)
                else:
                    out_buffer[index], out_buffer[index + 1], out_buffer[index + 2] = 0, 0, 0

        return pygame.image.frombuffer(out_buffer, (w, h), 'RGB'), out_buffer

    else:
        with nogil:
            for i in prange(0, b_length, 3, schedule=METHOD, num_threads=THREADS):
                # ITU-R BT.601 luma coefficients
                lum = c_buffer[i] * 0.299 + c_buffer[i+1] * 0.587 + c_buffer[i+2] * 0.114
                if lum > threshold:
                    c = (lum - threshold) / lum
                    if mode == 0:
                        # RGB
                        out_buffer[i    ] = <unsigned char>(c_buffer[i    ] * c)
                        out_buffer[i + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[i + 2] = <unsigned char>(c_buffer[i + 2] * c)
                    else:
                        # BGR
                        out_buffer[i    ] = <unsigned char>(c_buffer[i + 2] * c)
                        out_buffer[i + 1] = <unsigned char>(c_buffer[i + 1] * c)
                        out_buffer[i + 2] = <unsigned char>(c_buffer[i    ] * c)
                else:
                    out_buffer[i], out_buffer[i + 1], out_buffer[i + 2] = 0, 0, 0

        return pygame.image.frombuffer(out_buffer,(w, h), 'RGB'), out_buffer


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] bpf32_c(image, int threshold = 128):
    """
    Bright pass filter compatible 32-bit 
    
    Bright pass filter for 32-bit image (method using 3d array data structure)
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param image: pygame.Surface 32 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a 3d numpy.ndarray type (w, h, 4) filtered (only bright area of the image remains).
    """

    # Fallback to default threshold value if argument
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for argument image, got %s " % type(image)

    # make sure the surface is 32-bit format RGB
    if not image.get_bitsize() == 32:
        raise ValueError('Surface is not 32-bit format.')

    try:
        rgba_array = pygame.surfarray.pixels3d(image)
        alpha_ = pygame.surfarray.pixels_alpha(image)
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int w, h
    w, h = rgba_array.shape[:2]

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        unsigned char [:, :, :] rgba = rgba_array
        unsigned char [:, :, ::1] out_rgba = numpy.empty((w, h, 4), uint8)
        unsigned char [:, :] alpha = alpha_
        int i = 0, j = 0
        float lum, lum2, c

    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in prange(0, h):
                # ITU-R BT.601 luma coefficients
                lum = rgba[i, j, 0] * 0.299 + rgba[i, j, 1] * 0.587 + rgba[i, j, 2] * 0.114

                if lum > threshold:
                    c = (lum - threshold) / lum
                    out_rgba[i, j, 0] = <unsigned char>(rgba[i, j, 0] * c)
                    out_rgba[i, j, 1] = <unsigned char>(rgba[i, j, 1] * c)
                    out_rgba[i, j, 2] = <unsigned char>(rgba[i, j, 2] * c)
                    out_rgba[i, j, 3] = alpha[i, j]
                else:
                    out_rgba[i, j, 0], out_rgba[i, j, 1], \
                    out_rgba[i, j, 2], out_rgba[i, j, 3] = 0, 0, 0, 0

    return out_rgba


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf32_b_c(image, int threshold = 128):
    """
    Bright pass filter for 32-bit image (method using c-buffer)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param image: pygame.Surface 32 bit format (RGBA)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values
    :return: Return a 32-bit pygame.Surface filtered (only bright area of the image remains).
    """
 
    # Fallback to default threshold value if arguement
    # threshold value is incorrect
    if 0 > threshold > 255:
        printf("\nArgument threshold must be in range [0...255], fallback to default value 128.")
        threshold = 128    

    assert isinstance(image, pygame.Surface), \
           "\nExpecting pygame surface for arguement image, got %s " % type(image)

    cdef:
        int w, h
    w, h = image.get_size()

    # make sure the surface is 32-bit format RGBA
    if not image.get_bitsize() == 32:
        raise ValueError('Surface is not 32-bit format.')

    try:
        # BGRA buffer
        buffer_ = image.get_view('2')
        
    except (pygame.error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 4, schedule=METHOD, num_threads=THREADS):
            # ITU-R BT.601 luma coefficients
            lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
            if lum > threshold:
                c = (lum - threshold) / lum
                # BGRA to RGBA
                out_buffer[i    ] = <unsigned char>(cbuffer[i + 2  ] * c)
                out_buffer[i + 1] = <unsigned char>(cbuffer[i + 1  ] * c)
                out_buffer[i + 2] = <unsigned char>(cbuffer[i      ] * c)
                out_buffer[i + 3] = 255
            else:
                out_buffer[i], out_buffer[i+1], \
                out_buffer[i+2], out_buffer[i+3] = 0, 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGBA')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef filtering24_c(surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 24 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 24-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 24 bit
    """
    cdef int w, h, w_, h_
    w, h = surface_.get_size()
    try:
        w_, h_ = (<object>mask_).shape[:2]
    except (ValueError, pygame.error):
       raise ValueError(
           '\nArgument mask_ type not understood, '
           'expecting numpy.ndarray type (w, h) got %s ' % type(mask_))


    assert w == w_ and h == h_, \
        '\nSurface and mask size does not match (w:%s, h:%s), ' \
        '(w_:%s, h_:%s) ' % (w, h, w_, h_)

    try:
        rgb_ = pygame.surfarray.pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = pygame.surfarray.array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_.transpose(1, 0, 2)
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 3), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[j, i, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[j, i, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[j, i, 2] * mask[i, j])

    return pygame.image.frombuffer(rgb1, (w, h), 'RGB')



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef filtering32_c(surface_, mask_):
    """
    Multiply mask values with an array representing the surface pixels (Compatible 32 bit only).
    Mask values are floats in range (0 ... 1.0)

    :param surface_: pygame.Surface compatible 32-bit
    :param mask_: 2d array (MemoryViewSlice) containing alpha values (float).
    The mask_ output image is monochromatic (values range [0 ... 1.0] and R=B=G.
    :return: Return a pygame.Surface 32-bit
    """

    cdef int w, h, w_, h_
    w, h = surface_.get_size()

    try:
        w_, h_ = (<object>mask_).shape[:2]
    except (ValueError, pygame.error):
        raise ValueError(
            '\nArgument mask_ type not understood, expecting numpy.ndarray got %s ' % type(mask_))

    assert w == w_ and h == h_, 'Surface and mask size does not match.'

    try:
        rgb_ = pygame.surfarray.pixels3d(surface_)
    except (ValueError, pygame.error):
        try:
            rgb_ = pygame.surfarray.array3d(surface_)
        except (ValueError, pygame.error):
            raise ValueError('Incompatible surface.')

    cdef:
        unsigned char [:, :, :] rgb = rgb_
        unsigned char [:, :, ::1] rgb1 = numpy.empty((h, w, 4), numpy.uint8)
        float [:, :] mask = numpy.asarray(mask_, numpy.float32)
        int i, j
    with nogil:
        for i in prange(0, w, schedule=METHOD, num_threads=THREADS):
            for j in range(h):
                rgb1[j, i, 0] = <unsigned char>(rgb[i, j, 0] * mask[i, j])
                rgb1[j, i, 1] = <unsigned char>(rgb[i, j, 1] * mask[i, j])
                rgb1[j, i, 2] = <unsigned char>(rgb[i, j, 2] * mask[i, j])
                rgb1[j, i, 3] = <unsigned char>(mask[i, j] * 255.0)

    return pygame.image.frombuffer(rgb1, (w, h), 'RGBA')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef get_buffer(im_: pygame.Surface, view_mode_):
    """
    Return an object which exports a surface's internal pixel
    buffer as a C level array struct, Python level array interface
    or a C level buffer interface.
    
    :param im_: pygame.Surface
    :param view_mode_: mode
    :return :
    """
    try:
        buff = im_.get_view(view_mode_)
        
    except (pygame.error, ValueError) as e:
        print("\n%s " % e)
        if view_mode_ not in ("0", "1", "2", "3"):
            raise ValueError("Incorrect view_mode argument.")
        else:
            raise ValueError('Incorrect image format.')
              
    return buff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bloom_effect_buffer24_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 24 bit surface)
    This method is using C-buffer structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param surface_: pygame.Surface 24 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: integer; Number of Gaussian blur 5x5 to apply to downsided images.
    :return : Returns a pygame.Surface with a bloom effect (24 bit surface)


    """
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if not bitsize == 24:
        raise ValueError('\nIncorrect image format, expecting 24-bit got %s ' % bitsize)

    w2, h2   = w >> 1, h >> 1
    w4, h4   = w >> 2, h >> 2
    w8, h8   = w >> 3, h >> 3
    w16, h16 = w >> 4, h >> 4

    bpf_surface, bpf_array = bpf24_b_c(surface_, threshold=threshold_, transpose=False)

    # RESIZING BRIGHT PASS FILTER
    s2 = pygame.transform.scale(bpf_surface, (w2, h2))
    s4 = pygame.transform.scale(bpf_surface, (w4, h4))
    s8 = pygame.transform.scale(bpf_surface, (w8, h8))
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))

    b2 = pygame.image.tostring(s2, 'RGB')
    b2 = numpy.frombuffer(b2, dtype=numpy.uint8).copy()
    if smooth_ > 1:
        for r in range(smooth_):
            b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)
    else:
        b2_blurred, b2 = blur5x5_buffer24_c(b2, w2, h2, 3)

    b4 = pygame.image.tostring(s4, 'RGB')
    b4 = numpy.frombuffer(b4, dtype=numpy.uint8).copy()
    if smooth_ > 1:
        for r in range(smooth_):
            b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)
    else:
        b4_blurred, b4 = blur5x5_buffer24_c(b4, w4, h4, 3)

    b8 = pygame.image.tostring(s8, 'RGB')
    b8 = numpy.frombuffer(b8, dtype=numpy.uint8).copy()
    if smooth_ > 1:
        for r in range(smooth_):
            b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)
    else:
        b8_blurred, b8 = blur5x5_buffer24_c(b8, w8, h8, 3)

    b16 = pygame.image.tostring(s16, 'RGB')
    b16 = numpy.frombuffer(b16, dtype=numpy.uint8).copy()
    if smooth_ > 1:
        for r in range(smooth_):
            b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)
    else:
        b16_blurred, b16 = blur5x5_buffer24_c(b16, w16, h16, 3)

    s2  = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4  = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8  = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # if mask_ is not None:
    #     # Multiply mask surface pixels with mask values.
    #     # RGB pixels = 0 when mask value = 0.0, otherwise
    #     # modify RGB amplitude
    #     surface_cp = filtering24_c(surface_cp, mask_)
    return surface_cp




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bloom_effect_buffer32_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 32 bit surface)
    This method is using C-buffer structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c or bpf32_b_c (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param mask_: 2d array shape (w, h)  
    :param surface_: pygame.Surface 32 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: integer; Number of Gaussian blur 5x5 to apply to downsided images.
    :return : Returns a pygame.Surface with a bloom effect (32 bit surface)


    """
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "\nArgument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "\nArgument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    if not bitsize == 32:
        raise ValueError('\nIncorrect image format, expecting 32-bit got %s ' % bitsize)

    bpf_surface =  bpf32_b_c(surface_, threshold=threshold_)

    w2, h2 = w >> 1, h >> 1
    s2 = pygame.transform.scale(bpf_surface, (w2, h2))
    b2 = numpy.frombuffer(s2.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)
    else:
        b2_blurred, b2 = blur5x5_buffer32_c(b2, w2, h2, 4)#, mask_)

    # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    w4, h4 = w >> 2, h >> 2
    s4 = pygame.transform.scale(bpf_surface, (w4, h4))
    b4 = numpy.frombuffer(s4.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)
    else:
        b4_blurred, b4 = blur5x5_buffer32_c(b4, w4, h4, 4)#, mask_)

    # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    w8, h8 = w >> 3, h >> 3
    s8 = pygame.transform.scale(bpf_surface, (w8, h8))
    b8 = numpy.frombuffer(s8.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)
    else:
        b8_blurred, b8 = blur5x5_buffer32_c(b8, w8, h8, 4)#, mask_)

    # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    w16, h16 = w >> 4, h >> 4
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))
    b16 = numpy.frombuffer(s16.get_view("2"), numpy.uint8)
    if smooth_ > 1:
        for r in range(smooth_):
            b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)
    else:
        b16_blurred, b16 = blur5x5_buffer32_c(b16, w16, h16, 4)#, mask_)

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering32_c(surface_cp.convert_alpha(), mask_)

    return surface_cp



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bloom_effect_array24_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 24 bit surface)
    This method is using array structure.
    
    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf24_b_c  (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well. 
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.
    
    :param mask_:
    :param surface_: pygame.Surface 24-bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: Number of Gaussian blur 5x5 to apply to downside images.
    :return : Returns a pygame.Surface with a bloom effect (24 bit surface)


    """

    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "Argument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "Argument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    if not(bit_size==24):
        raise ValueError('Image format should be 24 bit, got %s ' % bit_size)


    w2, h2   = w >> 1, h >> 1
    w4, h4   = w >> 2, h >> 2
    w8, h8   = w >> 3, h >> 3
    w16, h16 = w >> 4, h >> 4
    # BRIGHT PASS FILTER
    bpf_surface, bpf_array = bpf24_c(surface_, threshold=threshold_, transpose=True)

    # RESIZING BRIGHT PASS FILTER
    s2  = pygame.transform.scale(bpf_surface, (w2, h2))
    s4  = pygame.transform.scale(bpf_surface, (w4, h4))
    s8  = pygame.transform.scale(bpf_surface, (w8, h8))
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))


    s2_array = numpy.array(s2.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
    if smooth_ > 1:
        for r in range(smooth_):
            s2_array = blur5x5_array24_c(s2_array)
    else:
        s2_array = blur5x5_array24_c(s2_array)
    b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGB')

    s4_array = numpy.array(s4.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
    if smooth_ > 1:
        for r in range(smooth_):
            s4_array = blur5x5_array24_c(s4_array)
    else:
        s4_array = blur5x5_array24_c(s4_array)
    b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGB')

    s8_array = numpy.array(s8.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
    if smooth_ > 1:
        for r in range(smooth_):
            s8_array = blur5x5_array24_c(s8_array)
    else:
        s8_array = blur5x5_array24_c(s8_array)
    b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGB')

    s16_array = numpy.array(s16.get_view('3'), dtype=numpy.uint8).transpose(1, 0, 2)
    if smooth_ > 1:
        for r in range(smooth_):
            s16_array = blur5x5_array24_c(s16_array)
    else:
        s16_array = blur5x5_array24_c(s16_array)
    b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGB')

    # METHOD USING scale_array24_c METHOD
    # downscale x 2 using fast scale pygame algorithm (no re-sampling)
    # w2, h2 = w >> 1, h >> 1
    # s2_array = scale_array24_c(bpf_array, w2, h2)
    # if smooth_ > 1:
    #     for r in range(smooth_):
    #         s2_array = blur5x5_array24_c(s2_array)
    # else:
    #     s2_array = blur5x5_array24_c(s2_array)
    # b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGB')
    # # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    # w4, h4 = w >> 2, h >> 2
    # s4_array = scale_array24_c(bpf_array, w4, h4)
    # if smooth_ > 1:
    #     for r in range(smooth_):
    #         s4_array = blur5x5_array24_c(s4_array)
    # else:
    #     s4_array = blur5x5_array24_c(s4_array)
    # b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGB')
    # # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    # w8, h8 = w >> 3, h >> 3
    # s8_array = scale_array24_c(bpf_array, w8, h8)
    # if smooth_ > 1:
    #     for r in range(smooth_):
    #         s8_array = blur5x5_array24_c(s8_array)
    # else:
    #     s8_array = blur5x5_array24_c(s8_array)
    # b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGB')
    # # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    # w16, h16 = w >> 4, h >> 4
    # s16_array = scale_array24_c(bpf_array, w16, h16)
    # if smooth_ > 1:
    #     for r in range(smooth_):
    #         s16_array = blur5x5_array24_c(s16_array)
    # else:
    #     s16_array = blur5x5_array24_c(s16_array)
    # b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGB')

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering24_c(surface_cp, mask_)

    return surface_cp


cdef bloom_effect_array32_c(surface_, int threshold_, int smooth_=1, mask_=None):
    """
    Create a bloom effect on a pygame.Surface (compatible 32 bit surface)
    This method is using array structure.

    definition:
        Bloom is a computer graphics effect used in video games, demos,
        and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.

    1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
      bpf32_b_c (adjust the threshold value to get the best filter effect).
    2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale method (no need to
      use smoothscale (bilinear filtering method).
    3)Apply a Gaussian blur 5x5 effect on each of the downsized bpf images (if smooth_ is > 1, then the Gaussian
      filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
    4)Re-scale all the bpf images using a bilinear filter (width and height of original image).
      Using an un-filtered rescaling method will pixelate the final output image.
      For best performances sets smoothscale acceleration.
      A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
      'SSE' allows SSE extensions as well.
    5)Blit all the bpf images on the original surface, use pygame additive blend mode for
      a smooth and brighter effect.

    Notes:
    The downscaling process of all sub-images could be done in a single process to increase performance.

    :param mask_:
    :param surface_: pygame.Surface 32-bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: Number of Gaussian blur 5x5 to apply to downsided images.
    :return : Returns a pygame.Surface with a bloom effect (24 bit surface)


    """
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "Argument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "Argument threshold_ must be in range [0...255] got %s " % threshold_

    cdef:
        int w, h, bit_size
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bit_size = surface_.get_bitsize()

    if not (bit_size==32):
        raise ValueError('Image format should be 32-bit, got %s ' % bit_size)

    bpf_array =  bpf32_c(surface_, threshold=threshold_)
    # downscale x 2 using fast scale pygame algorithm (no re-sampling)
    w2, h2 = w >> 1, h >> 1
    s2_array = scale_array32_c(bpf_array, w2, h2)
    if smooth_ > 1:
        for r in range(smooth_):
            s2_array = blur5x5_array32_c(s2_array)
    else:
        s2_array = blur5x5_array32_c(s2_array)
    b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGBA')
    # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    w4, h4 = w >> 2, h >> 2
    s4_array = scale_array32_c(bpf_array, w4, h4)
    if smooth_ > 1:
        for r in range(smooth_):
            s4_array = blur5x5_array32_c(s4_array)
    else:
        s4_array = blur5x5_array32_c(s4_array)
    b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGBA')
    # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    w8, h8 = w >> 3, h >> 3
    s8_array = scale_array32_c(bpf_array, w8, h8)
    if smooth_ > 1:
        for r in range(smooth_):
            s8_array = blur5x5_array32_c(s8_array)
    else:
        s8_array = blur5x5_array32_c(s8_array)
    b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGBA')
    # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    w16, h16 = w >> 4, h >> 4
    s16_array = scale_array32_c(bpf_array, w16, h16)
    if smooth_ > 1:
        for r in range(smooth_):
            s16_array = blur5x5_array32_c(s16_array)
    else:
        s16_array = blur5x5_array32_c(s16_array)
    b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGBA')

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    if mask_ is not None:
        # Multiply mask surface pixels with mask values.
        # RGB pixels = 0 when mask value = 0.0, otherwise
        # modify RGB amplitude
        surface_cp = filtering24_c(surface_cp, mask_)

    return surface_cp


DEF ONE_HALF      = 1.0 / 2.0
DEF ONE_FOURTH    = 1.0 / 4.0
DEF ONE_EIGHTH    = 1.0 / 8.0
DEF ONE_SIXTEENTH = 1.0 / 16.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_array24_mult_c(unsigned char [:, :, :] rgb_array):
    """
    MULTIPLE DOWNSCALING 
    DOWNSCALE/RESIZE SURFACE/ARRAY BY FACTOR 2, 4, 8, 16
    :param rgb_array: 3D array representing the surface type(width, height, 3) with unsigned char  
    :return: Return MEMORYVIEWSLICE; Returns input image downscale factor 2, 4, 8, 16 
    """

    cdef:
        int w1, h1, s

    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int w2  = <int>(w1 * ONE_HALF),   h2 = <int>(h1 * ONE_HALF)          # div 2
        int w4  = <int>(w1 * ONE_FOURTH), h4 = <int>(h1 * ONE_FOURTH)        # div 4
        int w8  = <int>(w1 * ONE_EIGHTH), h8 = <int>(h1 * ONE_EIGHTH)        # div 8
        int w16 = <int>(w1 * ONE_SIXTEENTH), h16 = <int>(h1 * ONE_SIXTEENTH) # div 16

    cdef:
        unsigned char [:, :, ::1] new_array_div2  = numpy.empty((w2, h2, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div4  = numpy.empty((w4, h4, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div8  = numpy.empty((w8, h8, 3), numpy.uint8)
        unsigned char [:, :, ::1] new_array_div16 = numpy.empty((w16, h16, 3), numpy.uint8)
        int x, y, xx2, xx4, xx8, xx16, yy2
        int r, g, b

    # TODO TEST WITH MAX AND MIN (INTEGER)
    with nogil:
        for x in prange(0, w1, schedule=METHOD, num_threads=THREADS):
            xx2  = <int>fmin(x * ONE_HALF,   w2-1)
            xx4  = <int>fmin(x * ONE_FOURTH, w4-1)
            xx8  = <int>fmin(x * ONE_EIGHTH, w8-1)
            xx16 = <int>fmin(x * ONE_SIXTEENTH, w16-1)
            for y in range(0, h1):

                yy2 = <int>fmin(y * ONE_HALF, h2-1)
                r, g, b = rgb_array[x, y, 0], rgb_array[x, y, 1], rgb_array[x, y, 2]
                new_array_div2[xx2, yy2, 0] = r
                new_array_div2[xx2, yy2, 1] = g
                new_array_div2[xx2, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_FOURTH, h4-1)
                new_array_div4[xx4, yy2, 0] = r
                new_array_div4[xx4, yy2, 1] = g
                new_array_div4[xx4, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_EIGHTH, h8-1)
                new_array_div8[xx8, yy2, 0] = r
                new_array_div8[xx8, yy2, 1] = g
                new_array_div8[xx8, yy2, 2] = b

                yy2 = <int>fmin(y * ONE_SIXTEENTH, h16-1)
                new_array_div16[xx16, yy2, 0] = r
                new_array_div16[xx16, yy2, 1] = g
                new_array_div16[xx16, yy2, 2] = b
    return new_array_div2, new_array_div4, new_array_div8, new_array_div16


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_alpha24_mult_c(unsigned char [:, :] alpha_array):
    """
    MULTIPLE DOWNSCALING 
    DOWNSCALE/RESIZE SURFACE/ARRAY BY FACTOR 2, 4, 8, 16
    :param alpha_array: 2D alpha array type(width, height) 
    :return: Return the input alpha array downscale factor 2, 4, 8, 16 (MEMORYVIEWSLICE)
    """
    cdef:
        int w1, h1

    try:
        w1, h1 = (<object>alpha_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        int w2  = <int>(w1 * ONE_HALF),   h2 = <int>(h1 * ONE_HALF)          # div 2
        int w4  = <int>(w1 * ONE_FOURTH), h4 = <int>(h1 * ONE_FOURTH)        # div 4
        int w8  = <int>(w1 * ONE_EIGHTH), h8 = <int>(h1 * ONE_EIGHTH)        # div 8
        int w16 = <int>(w1 * ONE_SIXTEENTH), h16 = <int>(h1 * ONE_SIXTEENTH) # div 16

    cdef:
        unsigned char [:, ::1] new_array_div2  = numpy.empty((w2, h2), numpy.uint8)
        unsigned char [:, ::1] new_array_div4  = numpy.empty((w4, h4), numpy.uint8)
        unsigned char [:, ::1] new_array_div8  = numpy.empty((w8, h8), numpy.uint8)
        unsigned char [:, ::1] new_array_div16 = numpy.empty((w16, h16), numpy.uint8)
        int x, y, xx2, xx4, xx8, xx16, yy2

    # TODO TEST WITH MAX AND MIN (INTEGER)
    with nogil:
        for x in prange(0, w1, schedule=METHOD, num_threads=THREADS):
            xx2  = <int>fmin(x * ONE_HALF,   w2-1)
            xx4  = <int>fmin(x * ONE_FOURTH, w4-1)
            xx8  = <int>fmin(x * ONE_EIGHTH, w8-1)
            xx16 = <int>fmin(x * ONE_SIXTEENTH, w16-1)
            for y in range(0, h1):

                yy2 = <int>fmin(y * ONE_HALF, h2-1)
                new_array_div2[xx2, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_FOURTH, h4-1)
                new_array_div4[xx4, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_EIGHTH, h8-1)
                new_array_div8[xx8, yy2] = alpha_array[x, y]

                yy2 = <int>fmin(y * ONE_SIXTEENTH, h16-1)
                new_array_div16[xx16, yy2] = alpha_array[x, y]

    return new_array_div2, new_array_div4, new_array_div8, new_array_div16



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef scale_alpha24_single_c(unsigned char [:, :] alpha_array, int w2, int h2):
    """
    RESCALE ALPHA ARRAY TYPE (W, H) TO DIMENSIONS (W2, H2)
    :param alpha_array: 2D Alpha array to rescale to w2, h2 dimensions 
    :param w2: final width
    :param h2: final height
    :return: Alpha array rescale to (w2, h2)
    """
    cdef:
        int w1, h1

    try:
        w1, h1 = (<object>alpha_array).shape[:2]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')


    cdef:
        unsigned char [:, ::1] new_array_div2  = numpy.empty((w2, h2), numpy.uint8)
        int x, y, xx, yy
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
    with nogil:
        for x in prange(0, w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(0, h2):
                yy = <int>(y * fy)
                new_array_div2[x, y] = alpha_array[xx, yy]

    return new_array_div2



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array24_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    Rescale a 24-bit format image from its given array 
    The final array is equivalent to the input array re-scale and transposed.
    
    :param rgb_array: RGB numpy.ndarray, format (w, h, 3) numpy.uint8
    :param w2: new width 
    :param h2: new height
    :return: Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 3) uint8
    """

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 3), numpy.uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[x, y, 0] = rgb_array[xx, yy, 0]
                new_array[x, y, 1] = rgb_array[xx, yy, 1]
                new_array[x, y, 2] = rgb_array[xx, yy, 2]

    return new_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array32_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    Rescale a 32-bit format image from its given array 
    The final array is equivalent to the input array re-scale and transposed.
    
    :param rgb_array: RGB numpy.ndarray, format (w, h, 4) numpy.uint8 with alpha channel
    :param w2: new width 
    :param h2: new height
    :return: Return a MemoryViewSlice 3d numpy.ndarray format (w, h, 4) uint8
    """

    cdef int w1, h1, s
    try:
        w1, h1, s = (<object>rgb_array).shape[:3]
    except (ValueError, pygame.error) as e:
        raise ValueError('\nArray shape not understood.')

    cdef:
        unsigned char [:, :, ::1] new_array = numpy.zeros((h2, w2, 4), numpy.uint8)
        float fx = <float>w1 / <float>w2
        float fy = <float>h1 / <float>h2
        int x, y, xx, yy
    with nogil:
        for x in prange(w2, schedule=METHOD, num_threads=THREADS):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[y, x, 0] = rgb_array[xx, yy, 0]
                new_array[y, x, 1] = rgb_array[xx, yy, 1]
                new_array[y, x, 2] = rgb_array[xx, yy, 2]
                new_array[y, x, 3] = rgb_array[xx, yy, 3]

    return new_array
