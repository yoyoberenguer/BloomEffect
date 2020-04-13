#cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True

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

# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    print("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")
    raise SystemExit


__version__ = 1.01

## VERSION 1.0
## Original version.

## VERSION 1.01
## Modify algorithms vmap_buffer_c, vfb_rgb_c, vfb_rgba_c, vfb_c
## in order to swap row and columns automatically. Previous version
## was waiting for a user input when calling those function by inverting
## row and columns manually.


DEF M_VOID  = 0
DEF M_BOOL  = 1
DEF M_BYTE  = 2
DEF M_UBYTE = 3
DEF M_SHORT = 4
DEF M_USHORT= 5
DEF M_INT   = 6
DEF M_UINT  = 7
DEF M_HALF  = 8
DEF M_FLOAT = 9
DEF M_DOUBLE= 10

cdef extern from 'C.c' nogil:
    struct m_image:
       void *data;
       int size;
       int width;
       int height;
       int comp;
       char type;

    void m_image_create(m_image *image, char type_, int width, int height, int comp)
    void m_image_destroy(m_image *image);
    void m_flip_buffer(m_image *src, m_image *dest);

# MAP BUFFER INDEX VALUE INTO 3D INDEXING
cpdef to3d(index, width, depth):
    return to3d_c(index, width, depth)

# MAP 3D INDEX VALUE INTO BUFFER INDEXING
cpdef to1d(x, y, z, width, depth):
    return to1d_c(x, y, z, width, depth)

# VERTICALLY FLIP A SINGLE BUFFER VALUE
cpdef vmap_buffer(index, width, height, depth):
    return vmap_buffer_c(index, width, height, depth)

# FLIP VERTICALLY A BUFFER (TYPE RGB)
cpdef vfb_rgb(source, target, width, height):
    return vfb_rgb_c(source, target, width, height)

# FLIP VERTICALLY A BUFFER (TYPE RGBA)
cpdef vfb_rgba(source, target, width, height):
    return vfb_rgba_c(source, target, width, height)

# FLIP VERTICALLY A BUFFER (TYPE ALPHA, (WIDTH, HEIGHT))
cpdef vfb(source, target, width, height):
    return vfb_c(source, target, width, height)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef testing_pure_c(unsigned char [:] buffer_, int width, int height):

    cdef:
        m_image foo1;
        m_image foo2;
        int b_length, r;

    b_length = len(buffer_)
    # Create two buffers
    m_image_create(&foo1, M_UINT, width, height, 3)
    m_image_create(&foo2, M_UINT, width, height, 3)

    foo1.data = &buffer_[0]
    foo2.data = &buffer_[0]

    m_flip_buffer(&foo1, &foo2)

    image = <unsigned char *>foo1.data
    return image


# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline xyz to3d_c(int index, int width, int depth)nogil:
    """
    Map a 1d buffer pixel values into a 3d array, e.g buffer[index] --> array[i, j, k]
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
cdef inline int to1d_c(int x, int y, int z, int width, int depth)nogil:
    """
    Map a 3d array value RGB(A) into a 1d buffer. e.g array[i, j, k] --> buffer[index]
   
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
cdef inline int vmap_buffer_c(int index, int width, int height, int depth)nogil:
    """
    Vertically flipped a single buffer value.
     
    :param index: integer; index value to convert
    :param width: integer; Original image width 
    :param height: integer; Original image height
    :param depth: integer; Original image depth=3 for RGB or 4 for RGBA
    :return: integer value pointing to the pixel in the buffer (traversed vertically). 
    """
    cdef:
        int ix
        int x, y, z
    ix = index // depth
    y = int(ix / width)
    x = ix % width
    z = index % depth
    return (x * height * depth) + (depth * y) + z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [:] vfb_rgb_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer type RGB
    
    Flip a C-buffer vertically filled with RGB values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGB otherwise a valuerror
    will be raised.
    SOURCE AND TARGET ARRAY MUST BE SAME SIZE.
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
    :param width: integer; Source array's width (or width of the original image) 
    :param height: integer; source array's height (or height of the original image)
    :return: Return a vertically flipped 1D RGB buffer (swapped rows and columns of the 2d model) 
    """
    cdef:
        int i, j, k, index
        unsigned char [:] flipped_array = target

    for i in prange(0, height * 3, 3):
        for j in range(0, width):
            index = i + (height * 3 * j)
            for k in range(3):
                flipped_array[(j * 3) + (i * width) + k] =  <unsigned char>source[index + k]

    return flipped_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [:] vfb_rgba_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    """
    Vertically flipped buffer
    
    Flip a C-buffer vertically filled with RGBA values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGBA otherwise a valuerror
    will be raised.
    SOURCE AND TARGET ARRAY MUST BE SAME SIZE.
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
    :param width: integer; Source array's width (or width of the original image) 
    :param height: integer; source array's height (or height of the original image)
    :return: Return a vertically flipped 1D RGBA buffer (swapped rows and columns of the 2d model) 
    """

    cdef:
        int i, j, k, index, v
        unsigned char [:] flipped_array = target

    for i in prange(0, height * 4, 4):
        for j in range(0, width):
            index = i + (height * 4 * j)
            v = (j * 4) + (i * width)
            for k in range(4):
                flipped_array[v + k] =  <unsigned char>source[index + k]

    return flipped_array

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [::1] vfb_c(
        unsigned char [:] source, unsigned char [::1] target, int width, int height)nogil:
    """
    Flip vertically the content (e.g alpha values) of an 1d buffer structure.
    buffer representing an array type (w, h) 
    
    :param source: 1d buffer created from array type(w, h) 
    :param target: 1d buffer numpy.empty(ax_ * ay_, dtype=numpy.uint8) that will be the equivalent 
    of the source array but flipped vertically 
    :param width: source width 
    :param height: source height
    :return: return 1d buffer (source array flipped)
    """
    cdef:
        int i, j
        unsigned char [::1] flipped_array = target

    for i in range(0, height, 1):
        for j in range(0, width, 1):
            flipped_array[j + (i * width)] =  <unsigned char>source[i + (height * j)]
    return flipped_array
