

# GAUSSIAN BLUR KERNEL 5x5 COMPATIBLE 24-32 BIT SURFACE
cdef blur5x5_buffer24_c(unsigned char [::1] rgb_buffer, int width, int height, int depth)
cdef blur5x5_buffer32_c(unsigned char [:] rgba_buffer, int width, int height, int depth)
cdef unsigned char [:, :, ::1] blur5x5_array24_c(unsigned char [:, :, :] rgb_array_)
cdef unsigned char [:, :, ::1] blur5x5_array32_c(unsigned char [:, :, :] rgb_array_)

# BRIGHT PASS FILTERS
cdef bpf24_c(image, int threshold=*, bint transpose=*)
cdef bpf24_b_c(image, int threshold=*, bint transpose=*)
cdef unsigned char [:, :, ::1] bpf32_c(image, int threshold=*)
cdef bpf32_b_c(image, int threshold=*)

# FILTERING
cdef filtering24_c(surface_, mask_)
cdef filtering32_c(surface_, mask_)

# BLOOM EFFECT
cdef bloom_effect_buffer24_c(surface_, int threshold_, int smooth_=*, mask_=*)
cdef bloom_effect_buffer32_c(surface_, int threshold_, int smooth_=*, mask_=*)
cdef bloom_effect_array24_c(surface_, int threshold_, int smooth_=*, mask_=*)
cdef bloom_effect_array32_c(surface_, int threshold_, int smooth_=*, mask_=*)

# RESCALE ARRAY
cdef scale_array24_mult_c(unsigned char [:, :, :] rgb_array)
cdef scale_alpha24_mult_c(unsigned char [:, :] alpha_array)
cdef scale_alpha24_single_c(unsigned char [:, :] alpha_array, int w2, int h2)
cdef unsigned char [:, :, ::1] scale_array24_c(unsigned char [:, :, :] rgb_array, int w2, int h2)
cdef unsigned char [:, :, ::1] scale_array32_c(unsigned char [:, :, :] rgb_array, int w2, int h2)
