
# ROLL BUFFER COMPATIBLE 24-BIT TEXTURE
cdef scroll_buffer24_c(unsigned char [:] buffer_, int w, int h, int dy, int dx)
# ROLL BUFFER COMPATIBLE 32-BIT TEXTURE
cdef scroll_buffer32_c(unsigned char [:] buffer_, int w, int h, int dy, int dx)
# ROLL ARRAY 3D TYPE (W, H, 3) NUMPY.UINT8
cdef unsigned char [:, :, :] scroll_array24_c(unsigned char[:, :, :] rgb_array_, int dy, int dx)
# ROLL ARRAY 3D TYPE (W, H, 4) NUMPY.UINT8
cdef unsigned char [:, :, :] scroll_array32_c(unsigned char [:, :, :] rgb_array, int dy, int dx)
# ROLL ARRAY (INPUT RGB + ALPHA)
cdef scroll_array32m_c(unsigned char [:, :, :] rgb_array,
                       unsigned char [:, :] alpha_array, int dy, int dx)
# ROLL TRANSPARENCY ONLY
cdef scroll_transparency_c(surface, int dy, int dx)
# ROLL ARRAY (lateral and vertical)
cdef scroll_surface24_c(surface, int dy, int dx)
# ROLL IMAGE 32-bit
cdef scroll_surface32_c(surface, int dy, int dx)

# USE NUMPY LIBRARY (NUMPY.ROLL METHOD)
cdef roll_surface_c(surface_, dx=*, dy=*)
cdef roll_array_c(array_, dx=*, dy=*)

# STACK RGB & ALPHA ARRAY VALUES,
cdef stack_object_c(unsigned char[:, :, :] rgb_array_,
                    unsigned char[:, :] alpha_, bint transpose=*)
# STACK RGB & ALPHA ARRAY VALUES
cdef unsigned char[:, :,::1] stack_mem_c(unsigned char[:, :, :] rgb_array_,
                                         unsigned char[:, :] alpha_, bint transpose=*)
# UN-STACK RGBA ARRAY VALUES
cdef unstack_object_c(unsigned char[:, :, :] rgba_array_)
# STACK RGB AND ALPHA BUFFERS
cdef unsigned char[::1] stack_buffer_c(rgb_array_, alpha_, int w, int h, bint transpose=*)
# UN-STACK/SPLIT RGBA BUFFER WITH RGBA INTO
cdef unstack_buffer_c(unsigned char [:] rgba_buffer_, int w, int h)

# TRANSPOSE ROWS AND COLUMNS ARRAY (W, H, 3)
cdef unsigned char[:, :, ::1] transpose24_c(unsigned char[:, :, :] rgb_array_)
# TRANSPOSE ROWS AND COLUMNS ARRAY (W, H, 4)
cdef unsigned char[:, :, ::1] transpose32_c(unsigned char[:, :, :] rgb_array_)
