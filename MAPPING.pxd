###cython: boundscheck=False, wraparound=False, nonecheck=False, optimize.use_switch=True
# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;
# MAP BUFFER INDEX VALUE INTO 3D INDEXING
cdef xyz to3d_c(int index, int width, int depth)nogil;
# MAP 3D INDEX VALUE INTO BUFFER INDEXING
cdef int to1d_c(int x, int y, int z, int width, int depth)nogil;
# VERTICALLY FLIP A SINGLE BUFFER VALUE
cdef int vmap_buffer_c(int index, int width, int height, int depth)nogil;
# FLIP VERTICALLY A BUFFER (TYPE RGB)
cdef unsigned char [:] vfb_rgb_c(
        unsigned char [:] source, unsigned char [:] target, int width, int height)nogil;
# FLIP VERTICALLY A BUFFER (TYPE RGBA)
cdef unsigned char [:] vfb_rgba_c(
        unsigned char [:] source, unsigned char [:] target, int width, int height)nogil;
# FLIP VERTICALLY A BUFFER (TYPE ALPHA, (WIDTH, HEIGHT))
cdef unsigned char [::1] vfb_c(
        unsigned char [:] source, unsigned char [::1] target, int width, int height)nogil;
