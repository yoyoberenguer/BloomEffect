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


"""

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

cimport numpy as np
from libc.math cimport sin, sqrt, cos, atan2, pi, round, floor, fmax, fmin, pi, tan, exp, ceil, fmod
from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX, qsort, malloc, free, abs
import timeit

# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline xyz to3d(int index, int width, int depth)nogil:
    """
    Map a 1d buffer pixel index value into a 3d array, e.g buffer[index] --> array[i, j, k]
    Both (buffer and array) must have the same length (width * height * depth)
    To speed up the process, no checks are performed upon the function call and
    index, width and depth values must be > 0.

    :param index: integer; Buffer index value
    :param width: integer; image width
    :param depth: integer; image depth (3)RGB, (4)RGBA
    :return: Array index/key [x][y][z] pointing to a pixel RGB(A) identical
    to the buffer index value. Array index values are placed into a C structure (xyz)
    """
    cdef xyz v;
    cdef int ix = index // depth
    v.y = <int>(ix / width)
    v.x = ix % width
    v.z = index % depth
    return v


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int to1d(int x, int y, int z, int width, int depth)nogil:
    """
    Map a 3d array index value RGB(A) into a 1d buffer. e.g array[i, j, k] --> buffer[index]
   
    To speed up the process, no checks are performed upon the function call and
    x, y, z, width and depth values must be > 0 and both (buffer and array) must
    have the same length (width * height * depth)
    
    :param x: integer; array row value   
    :param y: integer; array column value
    :param z: integer; RGB(3) or RGBA(4) 
    :param width: source image width 
    :param depth: integer; source image depth (3)RGB or (4)RGBA
    :return: return the index value into a buffer for the given 3d array indices [x][y][z]. 
    """
    return <int>(y * width * depth + x * depth + z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline int vmap_buffer(int index, int width, int height, int depth)nogil:
    """
    Vertically flipped a single buffer value.
     
    :param index: integer; index value 
    :param width: integer; image width
    :param height: integer; image height
    :param depth: integer; image depth (3)RGB or (4)RGBA
    :return: integer value pointing to the pixel in the buffer (traversed vertically). 
    """
    cdef:
        int ix
        int x, y, z
    ix = index // 4
    y = int(ix / height)
    x = ix % height
    z = index % depth
    return (x * width * depth) + (depth * y) + z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] vfb_rgb(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer
    
    Flip a C-buffer vertically filled with RGB values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGB otherwise a valuerror
    will be raised.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 27), pixel format RGB
    
    buffer = [RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, RGB9]
    Equivalent 3d model would be (3x3x3):
    3d model = [RGB1 RGB2 RGB3]
               [RGB4 RGB5 RGB6]
               [RGB7 RGB8 RGB9]

    After vbf_rgb:
    output buffer = [RGB1, RGB4, RGB7, RGB2, RGB5, RGB8, RGB3, RGB6, RGB9]
    and its equivalent 3d model
    3D model = [RGB1, RGB4, RGB7]
               [RGB2, RGB5, RGB8]
               [RGB3, RGB6, RGB9]
        
    :param source: 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGB. 
     
    :param target: Target buffer must have same length than source buffer)
    :param width: integer; width of the image 
    :param height: integer; height of the image
    :return: Return a vertically flipped buffer 
    """
    cdef:
        int i, j, k, index
        unsigned char [:] flipped_array = target

    for i in prange(0, width * 3, 3):
        for j in range(0, height):
            index = i + (width * 3 * j)
            for k in range(3):
                flipped_array[(j * 3) + (i * height) + k] =  <unsigned char>source[index + k]

    return flipped_array



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:] vfb_rgba(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer
    
    Flip a C-buffer vertically filled with RGBA values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGBA otherwise a valuerror
    will be raised.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 36), pixel format RGBA
    
    buffer = [RGBA1, RGBA2, RGBA3, RGBA4, RGBA5, RGBA6, RGBA7, RGBA8, RGBA9]
    Equivalent 3d model would be (3x3x4):
    3d model = [RGBA1 RGBA2 RGBA3]
               [RGBA4 RGBA5 RGBA6]
               [RGBA7 RGBA8 RGBA9]

    After vbf_rgba:
    output buffer = [RGB1A, RGB4A, RGB7A, RGB2A, RGB5A, RGBA8, RGBA3, RGBA6, RGBA9]
    and its equivalent 3d model
    3D model = [RGBA1, RGBA4, RGBA7]
               [RGBA2, RGBA5, RGBA8]
               [RGBA3, RGBA6, RGBA9]
        
    :param source: 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGBA. 
     
    :param target: Target buffer must have same length than source buffer)
    :param width: integer; width of the image 
    :param height: integer; height of the image
    :return: Return a vertically flipped buffer 
    """
    cdef:
        int i, j, k, index
        unsigned char [:] flipped_array = target

    for i in prange(0, width * 4, 4):
        for j in range(0, height):
            index = i + (width * 4 * j)
            for k in range(4):
                flipped_array[(j * 4) + (i * height) + k] =  <unsigned char>source[index + k]

    return flipped_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gaussian_blur5x5_buffer_24_c(rgb_buffer, int width, int height, int depth):
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
    :param rgb_buffer: 1d buffer representing a 24bit format pygame.Surface (type BufferProxy)
    :return: 24-bit Pygame.Surface without per-pixel information.
    """

    assert isinstance(rgb_buffer, pygame.BufferProxy),\
        'Positional argument rgb_buffer must be a BufferProxy, got %s ' % type(rgb_buffer)

    cdef:
        int b_length= rgb_buffer.length

    # Transform the rgb_buffer (transpose row and column)
    # and flatten the array into 1d buffer
    array_ = numpy.array(rgb_buffer, dtype=uint8).transpose(1, 0, 2)
    flat = array_.flatten(order='C')


    # check if the buffer length equal theoretical length
    if b_length != (width * height * depth):
        print("\nIncorrect 24-bit format image.")

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = \
        #     numpy.array(([1.0/16.0,
        #                   4.0/16.0,
        #                   6.0/16.0,
        #                   4.0/16.0,
        #                   1.0/16.0]), dtype=numpy.float32, copy=False)
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]

        short int kernel_half = 2
        short int kernel_length = len(kernel)
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
        unsigned char [::1] buffer_ = flat
        
    with nogil:
        # horizontal convolution
        # goes through all RGB values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule='static', num_threads=4):

            r, g, b = 0, 0, 0

            # v.x point to the row value of the equivalent 3d array (width, height, depth)
            # v.y point to the column value ...
            # v.z is always = 0 as the i value point always
            # to the red color of a pixel in the C-buffer structure
            v = to3d(i, width, depth)

            # testing
            # index = to1d(v.x, v.y, v.z, width, 3)
            # print(v.x, v.y, v.z, i, index)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                # Convert 1d indexing into a 3d indexing
                # v.x correspond to the row index value in a 3d array
                # v.x is always pointing to the red color of a pixel (see for i loop with
                # step = 3) in the C-buffer data structure.
                xx = v.x + kernel_offset

                # avoid buffer overflow
                # xx must always be in range [0 ... width]
                if xx < 0:
                    # re-convert the 3d indexing into 1d buffer indexing
                    index = to1d(0, v.y, v.z, width, depth)
                    red, green, blue = buffer_[index],\
                        buffer_[index + 1], buffer_[index + 2]

                # avoid buffer overflow
                # xx must always be in range [0 ... width]
                elif xx > (width - 1):
                    # re-convert the 3d indexing into 1d buffer indexing
                    index = to1d(width - 1, v.y, v.z, width, depth)
                    red, green, blue = buffer_[index],\
                        buffer_[index + 1], buffer_[index + 2]

                else:
                    # Convert the 3d indexing into 1d buffer indexing
                    # The index value must always point to a red pixel
                    # v.z = 0
                    index = to1d(xx, v.y, v.z, width, depth)

                    # load the color value from the current pixel
                    red = buffer_[index]
                    green = buffer_[index + 1]
                    blue = buffer_[index + 2]


                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # place the new RGB values into an empty array (convolve)
            convolve[i], convolve[i+1],\
            convolve[i+2] = <unsigned char>r, <unsigned char>g, <unsigned char>b

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.
        for i in prange(0, b_length, depth, schedule='static', num_threads=4):

                r, g, b = 0, 0, 0

                v = to3d(i, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0:
                        index = to1d(v.x, 0, v.z, width, depth)
                        ii = vmap_buffer(index, width, height, depth)
                        red, green, blue = buffer_[ii],\
                        buffer_[ii+1], buffer_[ii+2]

                    elif yy > (height-1):
                        index = to1d(v.x, height-1, v.z, width, depth)
                        ii = vmap_buffer(index, width, height, depth)
                        red, green, blue = buffer_[ii],\
                        buffer_[ii+1], buffer_[ii+2]

                    else:

                        ii = to1d(v.x, yy, v.z, width, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k

                convolved[i], convolved[i+1], convolved[i+2], \
                    = <unsigned char>r, <unsigned char>g, <unsigned char>b

    return pygame.image.frombuffer(convolved, (width, height), "RGB")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef gaussian_blur5x5_buffer_32_c(rgb_buffer, int width, int height, int depth):
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

    :param depth: integer; image depth (4)RGBA 
    :param height: integer; image height
    :param width:  integer; image width
    :param rgb_buffer: 1d buffer representing a 24-bit format pygame.Surface  
    :return: 32-bit Pygame.Surface with per-pixel information and pixel array (C-buffer) 
    """

    assert isinstance(rgb_buffer, pygame.BufferProxy),\
        'Positional arguement rgb_buffer must be a BufferProxy, got %s ' % type(rgb_buffer)

    cdef:
        int b_length= rgb_buffer.length

    if b_length != (width * height * depth):
        print("\nIncorrect 32-bit format image.")

    # kernel 5x5 separable
    cdef:
        # float [::1] kernel = \
        #     numpy.array(([1.0/16.0,
        #                   4.0/16.0,
        #                   6.0/16.0,
        #                   4.0/16.0,
        #                   1.0/16.0]), dtype=numpy.float32, copy=False)
        float[5] kernel = [1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0]
        short int kernel_half = 2
        short int kernel_length = len(kernel)
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
        unsigned char [::1] buffer_ = numpy.frombuffer(rgb_buffer, numpy.uint8)

    with nogil:
        # horizontal convolution
        # goes through all RGB(A) values of the buffer and apply the convolution
        for i in prange(0, b_length, depth, schedule='static', num_threads=4):

            r, g, b = 0, 0, 0

            v = to3d(i, width, depth)

            for kernel_offset in range(-kernel_half, kernel_half + 1):

                k = kernel[kernel_offset + kernel_half]

                xx = v.x + kernel_offset * depth

                if xx < 0:
                    index = to1d(0, v.y, v.z, width, depth)
                    red, green, blue = buffer_[index],\
                        buffer_[index + 1], buffer_[index + 2]

                elif xx > (width - 1):
                    index = to1d(width-1, v.y, v.z, width, depth)
                    red, green, blue = buffer_[index],\
                        buffer_[index+1], buffer_[index+2]

                else:
                    ii = i + kernel_offset * depth
                    red, green, blue = buffer_[ii],\
                        buffer_[ii+1], buffer_[ii+2]

                r = r + red * k
                g = g + green * k
                b = b + blue * k

            # empty convolve array inherit all the new pixels values
            convolve[i], convolve[i+1], convolve[i+2], \
            convolve[i+3] = <unsigned char>r, <unsigned char>g, <unsigned char>b, buffer_[i+3]

        # Vertical convolution
        # In order to vertically convolve the kernel, we have to re-order the index value
        # to fetch data vertically with the vmap_buffer function.

        for i in prange(0, b_length, depth, schedule='static', num_threads=4):

                r, g, b = 0, 0, 0

                v = to3d(i, width, depth)

                for kernel_offset in range(-kernel_half, kernel_half + 1):

                    k = kernel[kernel_offset + kernel_half]

                    yy = v.y + kernel_offset

                    if yy < 0:
                        index = to1d(v.x, 0, v.z, width, depth)
                        ii = vmap_buffer(index, width, height, depth)
                        red, green, blue = convolve[ii],\
                        convolve[ii+1], convolve[ii+2]

                    elif yy > (height-1):
                        index = to1d(v.x, height-1, v.z, width, depth)
                        ii = vmap_buffer(index, width, height, depth)
                        red, green, blue = convolve[ii],\
                        convolve[ii+1], convolve[ii+2]

                    else:
                        ii = i + kernel_offset * depth
                        ii = vmap_buffer(ii, width, height, depth)
                        red, green, blue = convolve[ii],\
                            convolve[ii+1], convolve[ii+2]

                    r = r + red * k
                    g = g + green * k
                    b = b + blue * k
                ii = vmap_buffer(i, width, height, depth)
                convolved[ii], convolved[ii+1], convolved[ii+2], \
                convolved[ii+3] = <unsigned char>r, <unsigned char>g, <unsigned char>b, buffer_[i+3]

    return pygame.image.frombuffer(convolved, (width, height), "RGBA")



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] gaussian_blur5x5_array_24_c(unsigned char [:, :, :] rgb_array_):
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
        for y in prange(0, h, schedule='static', num_threads=4):  # range [0..h-1)

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
        for x in prange(0,  w, schedule='static', num_threads=4):

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
cdef unsigned char [:, :, ::1] gaussian_blur5x5_array_32_c(unsigned char [:, :, :] rgb_array_):
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
        for y in prange(0, h, schedule='static', num_threads=4):

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
        for x in prange(0,  w, schedule='static', num_threads=4):

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
cdef unsigned char [:, :, ::1] bpf24_c(image, int threshold = 128):
    """
    Bright pass filter compatible 24-bit 
    
    Bright pass filter for 24bit image (method using 3d array data structure)
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
    :param image: pygame.Surface 24 bit format (RGB)  without per-pixel information
    :param threshold: integer; Threshold to consider for filtering pixels luminance values,
    default is 128 range [0..255] unsigned char (python integer)
    :return: Return a 3d numpy.ndarray format (w, h, 3) (only bright area of the image remains).
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
    except (pygame.Error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int w, h
    w, h = rgb_array.shape[:2]

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)

    cdef:
        unsigned char [:, :, :] rgb = rgb_array
        unsigned char [:, :, ::1] out_rgb= numpy.empty((h, w, 3), numpy.uint8)
        int i = 0, j = 0
        float lum, c

    with nogil:
        for i in prange(0, w, schedule='static', num_threads=4):
            for j in prange(0, h):
                # ITU-R BT.601 luma coefficients
                lum = rgb[i, j, 0] * 0.299 + rgb[i, j, 1] * 0.587 + rgb[i, j, 2] * 0.114
                if lum > threshold:
                    c = (lum - threshold) / lum
                    out_rgb[j, i, 0] = <unsigned char>(rgb[i, j, 0] * c)
                    out_rgb[j, i, 1] = <unsigned char>(rgb[i, j, 1] * c)
                    out_rgb[j, i, 2] = <unsigned char>(rgb[i, j, 2] * c)
                else:
                    out_rgb[j, i, 0], out_rgb[j, i, 1], out_rgb[j, i, 2] = 0, 0, 0

    return out_rgb



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef bpf24_b_c(image, int threshold = 128):
    """
    Bright pass filter for 24bit image (method using c-buffer)
    
    Calculate the luminance of every pixels and applied an attenuation c = lum2 / lum
    with lum2 = max(lum - threshold, 0) and
    lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
    The output image will keep only bright area. You can adjust the threshold value
    default 128 in order to get the desire changes.
    
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
        int w, h
    w, h = image.get_size()

    # make sure the surface is 24-bit format RGB
    if not image.get_bitsize() == 24:
        raise ValueError('Surface is not 24-bit format.')

    try:

        buffer_ = image.get_view('2')
        
    except (pygame.Error, ValueError):
        raise ValueError('\nInvalid surface.')

    # check sizes
    assert w>0 and h>0,\
        'Incorrect surface dimensions should be (w>0, h>0) got (w:%s, h:%s)' % (w, h)
    
    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 3, schedule='static', num_threads=4):
            # ITU-R BT.601 luma coefficients
            lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
            if lum > threshold:
                c = (lum - threshold) / lum
                out_buffer[i] = <unsigned char>(cbuffer[i] * c)
                out_buffer[i+1] = <unsigned char>(cbuffer[i+1] * c)
                out_buffer[i+2] = <unsigned char>(cbuffer[i+2] * c)
            else:
                out_buffer[i], out_buffer[i+1], out_buffer[i+2] = 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGB')


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
        for i in prange(0, w, schedule='static', num_threads=4):
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

        buffer_ = image.get_view('2')
        
    except (pygame.Error, ValueError):
        raise ValueError('\nInvalid surface.')

    cdef:
        int b_length = buffer_.length
        unsigned char [:] cbuffer = numpy.frombuffer(buffer_, numpy.uint8)
        unsigned char [::1] out_buffer = numpy.empty(b_length, numpy.uint8)
        int i = 0
        float lum, c

    with nogil:
        for i in prange(0, b_length, 4, schedule='static', num_threads=4):
            # ITU-R BT.601 luma coefficients
            lum = cbuffer[i] * 0.299 + cbuffer[i+1] * 0.587 + cbuffer[i+2] * 0.114
            if lum > threshold:

                c = (lum - threshold) / lum
                out_buffer[i] = <unsigned char>(cbuffer[i] * c)
                out_buffer[i+1] = <unsigned char>(cbuffer[i+1] * c)
                out_buffer[i+2] = <unsigned char>(cbuffer[i+2] * c)
                out_buffer[i+3] = 255
            else:
                out_buffer[i], out_buffer[i+1], \
                out_buffer[i+2], out_buffer[i+3] = 0, 0, 0, 0

    return pygame.image.frombuffer(out_buffer, (w, h), 'RGBA')



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
cpdef bloom_effect_buffer(surface_, int threshold_, int smooth_=1):
    """
    Create a bloom effect on a pygame.Surface (compatible 24-32 bit surface)
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
    
    :param surface_: pygame.Surface 24-32 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: integer; Number of Guaussian blur 5x5 to apply to downsided images. 
    :return : Returns a pygame.Surface with a bloom effect (24 or 32 bit surface)


    """
    # Create a copy of the pygame surface,
    # TODO make a copy subroutine as pygame.copy method is
    # very slow
    surface_cp = surface_.copy()

    assert smooth_ > 0, \
           "Argument smooth_ must be > 0, got %s " % smooth_
    assert -1 < threshold_ < 256, \
           "Argument threshold_ must be in range [0...255] got %s " % threshold_
    
    cdef:
        int w, h, bitsize
        int w2, h2, w4, h4, w8, h8, w16, h16

    w, h = surface_.get_size()
    bitsize = surface_.get_bitsize()

    # process 24-bit fornat image RGB
    if bitsize == 24:
         blurfunction_call = gaussian_blur5x5_buffer_24_c
         bpfunction_call = bpf24_b_c
         view_mode = "3"

    # process 32-bit image format RGBA
    elif bitsize == 32:
         blurfunction_call = gaussian_blur5x5_buffer_32_c
         bpfunction_call = bpf32_b_c
         view_mode = "2"

    else:
        raise ValueError('Incorrect image format.')
    
    bpf_surface =  bpfunction_call(surface_, threshold=threshold_)
    
    # downscale x 2 using fast scale pygame algorithm (no re-sampling)
    w2, h2 = w >> 1, h >> 1
    s2 = pygame.transform.scale(bpf_surface, (w2, h2))
    b2 = get_buffer(s2, view_mode)
    if smooth_ > 1:
        for r in range(smooth_):
            b2_blurred = blurfunction_call(b2, w2, h2, 4 if bitsize == 32 else 3)
            b2 = b2_blurred.get_view(view_mode)
    else:
        b2_blurred = blurfunction_call(b2, w2, h2, 4 if bitsize == 32 else 3)
    
    
    # downscale x 4 using fast scale pygame algorithm (no re-sampling)
    w4, h4 = w >> 2, h >> 2
    s4 = pygame.transform.scale(bpf_surface, (w4, h4))
    b4 = s4.get_view(view_mode)
    if smooth_ > 1:
        for r in range(smooth_):
            b4_blurred = blurfunction_call(b4, w4, h4, 4 if bitsize == 32 else 3)
            b4 = b4_blurred.get_view(view_mode)
    else:
        b4_blurred = blurfunction_call(b4, w4, h4, 4 if bitsize == 32 else 3)
    
    # downscale x 8 using fast scale pygame algorithm (no re-sampling)
    w8, h8 = w >> 3, h >> 3
    s8 = pygame.transform.scale(bpf_surface, (w8, h8))
    b8 = s8.get_view(view_mode)
    if smooth_ > 1:
        for r in range(smooth_):
            b8_blurred = blurfunction_call(b8, w8, h8, 4 if bitsize == 32 else 3)
            b8 = b8_blurred.get_view(view_mode)
    else:
        b8_blurred = blurfunction_call(b8, w8, h8, 4 if bitsize == 32 else 3)
    
    # downscale x 16 using fast scale pygame algorithm (no re-sampling)
    w16, h16 = w >> 4, h >> 4
    s16 = pygame.transform.scale(bpf_surface, (w16, h16))
    b16 = s16.get_view(view_mode)
    if smooth_ > 1:
        for r in range(smooth_):
            b16_blurred = blurfunction_call(b16, w16, h16, 4 if bitsize == 32 else 3)
            b16 = b16_blurred.get_view(view_mode)
    else:
        b16_blurred = blurfunction_call(b16, w16, h16, 4 if bitsize == 32 else 3)
    
    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    
    return surface_cp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef bloom_effect_array(surface_, int threshold_, int smooth_=1):
    """
    Create a bloom effect on a pygame.Surface (compatible 24-32 bit surface)
    This method is using array structure.
    
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
    
    :param surface_: pygame.Surface 24-32 bit format surface
    :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
    :param smooth_: Number of Guaussian blur 5x5 to apply to downsided images. 
    :return : Returns a pygame.Surface with a bloom effect (24 or 32 bit surface)


    """
    # Create a copy of the pygame surface,
    # TODO make a copy subroutine as pygame.copy method is
    # very slow
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

    if bit_size not in (24, 32):
        raise ValueError('Incorrect image format.')

    if bit_size == 24:

        bpf_array =  bpf24_c(surface_, threshold=threshold_)
        # downscale x 2 using fast scale pygame algorithm (no re-sampling)
        w2, h2 = w >> 1, h >> 1
        s2_array = scale_array24_c(bpf_array, w2, h2)
        if smooth_ > 1:
            for r in range(smooth_):
                s2_array = gaussian_blur5x5_array_24_c(s2_array)
        else:
            s2_array = gaussian_blur5x5_array_24_c(s2_array)
        b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGB')
        # downscale x 4 using fast scale pygame algorithm (no re-sampling)
        w4, h4 = w >> 2, h >> 2
        s4_array = scale_array24_c(bpf_array, w4, h4)
        if smooth_ > 1:
            for r in range(smooth_):
                s4_array = gaussian_blur5x5_array_24_c(s4_array)
        else:
            s4_array = gaussian_blur5x5_array_24_c(s4_array)
        b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGB')
        # downscale x 8 using fast scale pygame algorithm (no re-sampling)
        w8, h8 = w >> 3, h >> 3
        s8_array = scale_array24_c(bpf_array, w8, h8)
        if smooth_ > 1:
            for r in range(smooth_):
                s8_array = gaussian_blur5x5_array_24_c(s8_array)
        else:
            s8_array = gaussian_blur5x5_array_24_c(s8_array)
        b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGB')
        # downscale x 16 using fast scale pygame algorithm (no re-sampling)
        w16, h16 = w >> 4, h >> 4
        s16_array = scale_array24_c(bpf_array, w16, h16)
        if smooth_ > 1:
            for r in range(smooth_):
                s16_array = gaussian_blur5x5_array_24_c(s16_array)
        else:
            s16_array = gaussian_blur5x5_array_24_c(s16_array)
        b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGB')

    else:
        bpf_array =  bpf32_c(surface_, threshold=threshold_)
        # downscale x 2 using fast scale pygame algorithm (no re-sampling)
        w2, h2 = w >> 1, h >> 1
        s2_array = scale_array32_c(bpf_array, w2, h2)
        if smooth_ > 1:
            for r in range(smooth_):
                s2_array = gaussian_blur5x5_array_32_c(s2_array)
        else:
            s2_array = gaussian_blur5x5_array_32_c(s2_array)
        b2_blurred = pygame.image.frombuffer(s2_array, (w2, h2), 'RGBA')
        # downscale x 4 using fast scale pygame algorithm (no re-sampling)
        w4, h4 = w >> 2, h >> 2
        s4_array = scale_array32_c(bpf_array, w4, h4)
        if smooth_ > 1:
            for r in range(smooth_):
                s4_array = gaussian_blur5x5_array_32_c(s4_array)
        else:
            s4_array = gaussian_blur5x5_array_32_c(s4_array)
        b4_blurred = pygame.image.frombuffer(s4_array, (w4, h4), 'RGBA')
        # downscale x 8 using fast scale pygame algorithm (no re-sampling)
        w8, h8 = w >> 3, h >> 3
        s8_array = scale_array32_c(bpf_array, w8, h8)
        if smooth_ > 1:
            for r in range(smooth_):
                s8_array = gaussian_blur5x5_array_32_c(s8_array)
        else:
            s8_array = gaussian_blur5x5_array_32_c(s8_array)
        b8_blurred = pygame.image.frombuffer(s8_array, (w8, h8), 'RGBA')
        # downscale x 16 using fast scale pygame algorithm (no re-sampling)
        w16, h16 = w >> 4, h >> 4
        s16_array = scale_array32_c(bpf_array, w16, h16)
        if smooth_ > 1:
            for r in range(smooth_):
                s16_array = gaussian_blur5x5_array_32_c(s16_array)
        else:
            s16_array = gaussian_blur5x5_array_32_c(s16_array)
        b16_blurred = pygame.image.frombuffer(s16_array, (w16, h16), 'RGBA')

    s2 = pygame.transform.smoothscale(b2_blurred, (w , h))
    s4 = pygame.transform.smoothscale(b4_blurred, (w , h))
    s8 = pygame.transform.smoothscale(b8_blurred, (w, h))
    s16 = pygame.transform.smoothscale(b16_blurred, (w, h))

    surface_cp.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    surface_cp.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    return surface_cp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unsigned char [:, :, ::1] scale_array24_c(unsigned char [:, :, :] rgb_array, int w2, int h2):
    """
    Rescale a 24-bit format image from its given array 
    
    :param rgb_array: RGB numpy.ndarray, format (w, h, 3) numpy.uint8
    :param w2: new width 
    :param h2: new height
    :return: Return a 3d numpy.ndarray format (w, h, 3) uint8
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
        for x in prange(w2, schedule='static', num_threads=4):
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
    
    :param rgb_array: RGB numpy.ndarray, format (w, h, 4) numpy.uint8 with alpha channel
    :param w2: new width 
    :param h2: new height
    :return: Return a 3d numpy.ndarray format (w, h, 4) uint8
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
        for x in prange(w2, schedule='static', num_threads=4):
            xx = <int>(x * fx)
            for y in range(h2):
                yy = <int>(y * fy)
                new_array[y, x, 0] = rgb_array[xx, yy, 0]
                new_array[y, x, 1] = rgb_array[xx, yy, 1]
                new_array[y, x, 2] = rgb_array[xx, yy, 2]
                new_array[y, x, 3] = rgb_array[xx, yy, 3]

    return new_array
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef bloom_effect_array(surface_, threshold_, smooth_=1):
#     """
#     Create a bloom effect on a pygame.Surface (compatible 24-32 bit surface)
#     This method is using 3d array data structures.
#
#     definition:
#         Bloom is a computer graphics effect used in video games, demos,
#         and high dynamic range rendering to reproduce an imaging artifact of real-world cameras.
#
#     1)First apply a bright pass filter to the pygame surface(SDL surface) using methods
#       bpf24_c or bpf32_c (adjust the threshold value to get the best filter effect).
#     2)Downside the newly created bpf image by factor x2, x4, x8, x16 using the pygame scale(no need to
#       use the method smoothscale (bilinear filtering method).
#     3)Apply a Gaussian blur 5x5 effect on each of the downsided bpf images (if smooth_ if > 1, then the Gaussian
#       filter 5x5 will by applied more than once. Note, this have little effect on the final image quality.
#     4)Re-scale all the bpf images using a bilinear filter (width and height of original image). Using
#       an un-filtered rescaling method will pixelate the final output image.
#       For best performances sets smoothscale acceleration.
#       A value of 'GENERIC' turns off acceleration. 'MMX' uses MMX instructions only.
#       'SSE' allows SSE extensions as well.
#     5)Blit all the bpf images on the original surface, use pygame additive blend mode for
#       a smooth and brighter effect.
#
#     Notes:
#     The downscaling process of all sub-images could be done in a single process to increase performance.
#
#     :param surface_: pygame.Surface 24-32 bit format surface
#     :param threshold_: integer; Threshold value used by the bright pass algorithm (default 128)
#     :param smooth_: Number of Guaussian blur 5x5 to apply to downsided images.
#     :return : Returns a pygame.Surface with a bloom effect (24 or 32 bit surface)
#
#
#     """
#
#     assert smooth_ > 0, \
#            "Argument smooth_ must be > 0, got %s " % smooth_
#     assert -1 < threshold_ < 256, \
#            "Argument threshold_ must be in range [0...255] got %s " % threshold_
#
#     cdef:
#         int w, h, bitsize
#         # '2' returns a (surface-width, surface-height) array of raw pixels.
#         # The pixels are surface-bytesize-d unsigned integers.
#         # The pixel format is surface specific.
#         # The 3 byte unsigned integers of 24 bit surfaces are unlikely accepted
#         # by anything other than other pygame functions.
#         char view_mode = "2"
#
#     w, h = surface_.get_size()
#
#
#     original_image = surface_.copy()
#
#     bitsize = surface_.get_bitsize()
#
#     # process 24-bit fornat image RGB
#     if bitsize == 24:
#         blurfunction_call = gaussian_blur5x5_array_24_c
#         bpfunction_call = bpf24_c
#
#     # process 32-bit image format RGBA
#     elif bitsize == 32:
#         blurfunction_call = gaussian_blur5x5_array_32_c
#         bpfunction_call = bpf32_c
#
#     else:
#         raise ValueError('Incorrect image format.')
#
#     surface_ =  bpfunction_call(surface_, threshold=threshold_)
#
#     # downscale x 2 using fast scale pygame algorithm (no re-sampling)
#     w2, h2 = w >> 1, h >> 1
#     s2 = pygame.transform.scale(surface_, (w2, h2))
#     b2_blurred_array = pygame.surfarray.pixels3d(s2)
#     if bitsize == 32:
#         b2_blurred_alpha = pygame.surfarray.pixels_alpha(s2)
#     else:
#         b2_blurred_alpha = None
#     if smooth_ > 1:
#         for r in range(smooth_):
#             b2_blurred_surface, b2_blurred_array = \
#                 blurfunction_call(b2_blurred_array, b2_blurred_alpha)
#     else:
#         b2_blurred_surface, b2_blurred_array = \
#             blurfunction_call(b2_blurred_array, b2_blurred_alpha)
#
#     # downscale x 4 using fast scale pygame algorithm (no re-sampling)
#     w4, h4 = w >> 2, h >> 2
#     s4 = pygame.transform.scale(surface_, (w4, h4))
#     b4_blurred_array = pygame.surfarray.pixels3d(s4)
#     if bitsize == 32:
#         b4_blurred_alpha = pygame.surfarray.pixels_alpha(s4)
#     else:
#         b4_blurred_alpha = None
#     if smooth_ > 1:
#         for r in range(smooth_):
#             b4_blurred_surface, b4_blurred_array =\
#                 blurfunction_call(b4_blurred_array, b4_blurred_alpha)
#     else:
#         b4_blurred_surface, b4_blurred_array = \
#             blurfunction_call(b4_blurred_array, b4_blurred_alpha)
#
#     # downscale x 8 using fast scale pygame algorithm (no re-sampling)
#     w8, h8 = w >> 3, h >> 3
#     s8 = pygame.transform.scale(surface_, (w8, h8))
#     b8_blurred_array = pygame.surfarray.pixels3d(s8)
#     if bitsize == 32:
#         b8_blurred_alpha = pygame.surfarray.pixels_alpha(s8)
#     else:
#         b8_blurred_alpha = None
#     if smooth_ > 1:
#         for r in range(smooth_):
#             b8_blurred_surface, b8_blurred_array =\
#                 blurfunction_call(b8_blurred_array, b8_blurred_alpha)
#     else:
#         b8_blurred_surface, b8_blurred_array =\
#             blurfunction_call(b8_blurred_array, b8_blurred_alpha)
#
#     # downscale x 16 using fast scale pygame algorithm (no re-sampling)
#     w16, h16 = w >> 4, h >> 4
#     s16 = pygame.transform.scale(surface_, (w16, h16))
#     b16_blurred_array = pygame.surfarray.pixels3d(s16)
#     if bitsize == 32:
#         b16_blurred_alpha = pygame.surfarray.pixels_alpha(s16)
#     else:
#         b16_blurred_alpha = None
#     if smooth_ > 1:
#         for r in range(smooth_):
#             b16_blurred_surface, b16_blurred_array =\
#                 blurfunction_call(b16_blurred_array, b16_blurred_alpha)
#     else:
#         b16_blurred_surface, b16_blurred_array =\
#             blurfunction_call(b16_blurred_array, b16_blurred_alpha)
#
#     s2 = pygame.transform.smoothscale(b2_blurred_surface, (w , h))
#     s4 = pygame.transform.smoothscale(b4_blurred_surface, (w , h))
#     s8 = pygame.transform.smoothscale(b8_blurred_surface, (w, h))
#     s16 = pygame.transform.smoothscale(b16_blurred_surface, (w, h))
#
#     original_image.blit(s2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
#     original_image.blit(s4, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
#     original_image.blit(s8, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
#     original_image.blit(s16, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
#
#     return original_image # , s2, s4, s8, s16


#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef test1(rgb_array, kwargs):
#     """
#     # Gaussian kernel 5x5
#         # |1   4   6   4  1|
#         # |4  16  24  16  4|
#         # |6  24  36  24  6|  x 1/256
#         # |4  16  24  16  4|
#         # |1  4    6   4  1|
#     This method is using convolution property and process the image in two passes,
#     first the horizontal convolution and last the vertical convolution
#     pixels convoluted outside image edges will be set to adjacent edge value
#
#     :param kwargs:
#     :param rgb_array: numpy.ndarray type (w, h, 3) uint8
#     :return: Return 24-bit pygame.Surface and numpy.ndarray type (w, h, 3) uint8
#     """
#
#     assert isinstance(rgb_array, numpy.ndarray),\
#         'Positional arguement rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)
#
#     cdef int w, h, dim
#     try:
#         w, h, dim = rgb_array.shape[:3]
#
#     except (ValueError, pygame.error) as e:
#         raise ValueError('\nArray shape not understood.')
#
#     assert w!=0 or h !=0, 'Array with incorrect shapes (w>0, h>0) got (%s, %s) ' % (w, h)
#
#     kernel_ = numpy.array(([1.0 / 16.0,
#                            4.0 / 16.0,
#                            6.0 / 16.0,
#                            4.0 / 16.0,
#                            1.0 / 16.0]), dtype=float32, copy=False)
#
#     # kernel 5x5 separable
#     cdef:
#         float [:] kernel = kernel_
#         short int kernel_half = 2
#         unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
#         unsigned char [:, :, ::1] convolved = numpy.empty((h, w, 3), dtype=uint8)
#         unsigned char [:, :, :] rgb_array_ = rgb_array
#         short int kernel_length = len(kernel)
#         int x, y, xx, yy, sr, sg, sb
#         float k, r, g, b
#         char kernel_offset
#         unsigned char red, green, blue
#
#     with nogil:
#         # horizontal convolution
#         for y in prange(0, h, schedule='static', num_threads=4):  # range [0..h-1)
#
#             for x in range(0, w):  # range [0..w-1]
#
#                 r, g, b = 0, 0, 0
#
#                 if 2 < x < w-2:
#                     sr = rgb_array_[x-2, y, 0] + rgb_array_[x-1, y, 0] + rgb_array_[x, y, 0] + \
#                     rgb_array_[x+1, y, 0] + rgb_array_[x+2, y, 0]
#                     sg = rgb_array_[x-2, y, 1] + rgb_array_[x-1, y, 1] + rgb_array_[x, y, 1] + \
#                     rgb_array_[x+1, y, 1] + rgb_array_[x+2, y, 1]
#                     sb = rgb_array_[x-2, y, 2] + rgb_array_[x-1, y, 2] + rgb_array_[x, y, 2] + \
#                     rgb_array_[x+1, y, 2] + rgb_array_[x+2, y, 2]
#
#                 if sr ==0 and sg == 0 and sb == 0:
#                     continue
#
#                 for kernel_offset in range(-kernel_half, kernel_half + 1):
#
#                     k = kernel[kernel_offset + kernel_half]
#
#                     xx = x + kernel_offset
#
#                     # check boundaries.
#                     # Fetch the edge pixel for the convolution
#                     if xx < 0:
#                         red, green, blue = rgb_array_[0, y, 0],\
#                         rgb_array_[0, y, 1], rgb_array_[0, y, 2]
#                     elif xx > (w - 1):
#                         red, green, blue = rgb_array_[w-1, y, 0],\
#                         rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
#                     else:
#                         red, green, blue = rgb_array_[xx, y, 0],\
#                             rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
#
#                     r = r + red * k
#                     g = g + green * k
#                     b = b + blue * k
#
#                 convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
#                     <unsigned char>g, <unsigned char>b
#
#         # Vertical convolution
#         for x in prange(0,  w, schedule='static', num_threads=4):
#
#             for y in range(0, h):
#                 r, g, b = 0, 0, 0
#
#                 if 2 < y < h-2:
#                     sr = rgb_array_[x, y-2, 0] + rgb_array_[x, y-1, 0] + rgb_array_[x, y, 0] + \
#                     rgb_array_[x, y+1, 0] + rgb_array_[x, y+2, 0]
#                     sg = rgb_array_[x, y-2, 1] + rgb_array_[x, y-1, 1] + rgb_array_[x, y, 1] + \
#                     rgb_array_[x, y+1, 1] + rgb_array_[x, y+2, 1]
#                     sb = rgb_array_[x, y-2, 2] + rgb_array_[x, y-1, 2] + rgb_array_[x, y, 2] + \
#                     rgb_array_[x, y+1, 2] + rgb_array_[x, y+2, 2]
#
#                 if sr ==0 and sg == 0 and sb == 0:
#                     continue
#
#                 for kernel_offset in range(-kernel_half, kernel_half + 1):
#
#                     k = kernel[kernel_offset + kernel_half]
#                     yy = y + kernel_offset
#
#                     if yy < 0:
#                         red, green, blue = convolve[x, 0, 0],\
#                         convolve[x, 0, 1], convolve[x, 0, 2]
#                     elif yy > (h -1):
#                         red, green, blue = convolve[x, h-1, 0],\
#                         convolve[x, h-1, 1], convolve[x, h-1, 2]
#                     else:
#                         red, green, blue = convolve[x, yy, 0],\
#                             convolve[x, yy, 1], convolve[x, yy, 2]
#
#                     r = r + red * k
#                     g = g + green * k
#                     b = b + blue * k
#
#                 convolved[y, x, 0], convolved[y, x, 1], convolved[y, x, 2] = \
#                     <unsigned char>r, <unsigned char>g, <unsigned char>b
#
#     return pygame.image.frombuffer(convolved, (w, h), 'RGB'), asarray(convolved)

#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef test2(rgb_array, kwargs):
#     """
#     # Gaussian kernel 5x5
#         # |1   4   6   4  1|
#         # |4  16  24  16  4|
#         # |6  24  36  24  6|  x 1/256
#         # |4  16  24  16  4|
#         # |1  4    6   4  1|
#     This method is using convolution property and process the image in two passes,
#     first the horizontal convolution and last the vertical convolution
#     pixels convoluted outside image edges will be set to adjacent edge value
#
#     :param kwargs:
#     :param rgb_array: numpy.ndarray type (w, h, 3) uint8
#     :return: Return 24-bit pygame.Surface and numpy.ndarray type (w, h, 3) uint8
#     """
#
#     assert isinstance(rgb_array, numpy.ndarray),\
#         'Positional arguement rgb_array must be a numpy.ndarray, got %s ' % type(rgb_array)
#
#     cdef int w, h, dim
#     try:
#         w, h, dim = rgb_array.shape[:3]
#
#     except (ValueError, pygame.error) as e:
#         raise ValueError('\nArray shape not understood.')
#
#     assert w!=0 or h !=0, 'Array with incorrect shapes (w>0, h>0) got (%s, %s) ' % (w, h)
#
#     kernel_ = numpy.array(([1.0 / 16.0,
#                            4.0 / 16.0,
#                            6.0 / 16.0,
#                            4.0 / 16.0,
#                            1.0 / 16.0]), dtype=float32, copy=False)
#
#     # kernel 5x5 separable
#     cdef:
#         float [:] kernel = kernel_
#         short int kernel_half = 2
#         unsigned char [:, :, ::1] convolve = numpy.empty((w, h, 3), dtype=uint8)
#         unsigned char [:, :, ::1] convolved = numpy.empty((h, w, 3), dtype=uint8)
#         unsigned char [:, :, :] rgb_array_ = rgb_array
#         short int kernel_length = len(kernel)
#         int x, y, xx, yy
#         float k, r, g, b, s
#         char kernel_offset
#         unsigned char red, green, blue
#
#     with nogil:
#         # horizontal convolution
#         for y in prange(0, h, schedule='static', num_threads=4):  # range [0..h-1)
#
#             for x in range(0, w):  # range [0..w-1]
#
#                 r, g, b = 0, 0, 0
#
#                 for kernel_offset in range(-kernel_half, kernel_half + 1):
#
#                     k = kernel[kernel_offset + kernel_half]
#
#                     xx = x + kernel_offset
#
#                     # check boundaries.
#                     # Fetch the edge pixel for the convolution
#                     if xx < 0:
#                         red, green, blue = rgb_array_[0, y, 0],\
#                         rgb_array_[0, y, 1], rgb_array_[0, y, 2]
#                     elif xx > (w - 1):
#                         red, green, blue = rgb_array_[w-1, y, 0],\
#                         rgb_array_[w-1, y, 1], rgb_array_[w-1, y, 2]
#                     else:
#                         red, green, blue = rgb_array_[xx, y, 0],\
#                             rgb_array_[xx, y, 1], rgb_array_[xx, y, 2]
#
#                     r = r + red * k
#                     g = g + green * k
#                     b = b + blue * k
#
#                 convolve[x, y, 0], convolve[x, y, 1], convolve[x, y, 2] = <unsigned char>r,\
#                     <unsigned char>g, <unsigned char>b
#
#         # Vertical convolution
#         for x in prange(0,  w, schedule='static', num_threads=4):
#
#             for y in range(0, h):
#                 r, g, b = 0, 0, 0
#
#                 for kernel_offset in range(-kernel_half, kernel_half + 1):
#
#                     k = kernel[kernel_offset + kernel_half]
#                     yy = y + kernel_offset
#
#                     if yy < 0:
#                         red, green, blue = convolve[x, 0, 0],\
#                         convolve[x, 0, 1], convolve[x, 0, 2]
#                     elif yy > (h -1):
#                         red, green, blue = convolve[x, h-1, 0],\
#                         convolve[x, h-1, 1], convolve[x, h-1, 2]
#                     else:
#                         red, green, blue = convolve[x, yy, 0],\
#                             convolve[x, yy, 1], convolve[x, yy, 2]
#
#                     r = r + red * k
#                     g = g + green * k
#                     b = b + blue * k
#
#                 convolved[y, x, 0], convolved[y, x, 1], convolved[y, x, 2] = \
#                     <unsigned char>r, <unsigned char>g, <unsigned char>b
#
#
#     return pygame.image.frombuffer(convolved, (w, h), 'RGB'), asarray(convolved)



#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# @cython.cdivision(True)
# cpdef scale_24b_c(surface, int w1, int h1, int w2, int h2):
#     """
#     Rescale a given surface
#
#     :param surface: pygame.Surface to rescale, compatible 24-32 bit surface.
#         width and height must be >0 otherwise raise a value error.
#     :param w2: width for new surface
#     :param h2: height for new surface
#     :return: return a rescale pygame.Surface 24-bit without per-pixel
#         transparency, dimensions (w2, h2).
#     """
#
#     try:
#         buffer_ = surface.get_view('3')
#         array_ = numpy.array(buffer_, dtype=uint8).transpose(1, 0, 2)
#         flat = array_.flatten(order='C')
#
#     except (pygame.error, ValueError):
#         raise ValueError('\nIncompatible pixel format.')
#
#     cdef:
#         int b_length = buffer_.length
#         int new_length = w2 * h2 * 3
#         unsigned char [::1] new_array = numpy.empty(new_length, numpy.uint8)
#         unsigned char [:] buff = flat
#         float fx = <float>w1 / <float>w2
#         float fy = <float>h1 / <float>h2
#         int xx, yy, i, index
#         xyz v
#
#     with nogil:
#         for i in prange(0, new_length, 3):
#             v = to3d(i, w2, h2)
#             xx = <int>(v.x * fx)
#             yy = <int>(v.y * fy)
#             index = to1d(xx, yy, 0, w1, 3)
#             index = vmap_buffer(index, w2, h2, 3)
#             new_array[i] = buff[index]
#             new_array[i+1] = buff[index+1]
#             new_array[i+2] = buff[index+2]
#
#     return pygame.image.frombuffer(new_array, (w2, h2), 'RGB')
