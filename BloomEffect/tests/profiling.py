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
# PYGAME IS REQUIRED
import numpy

try:
    import pygame

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
                      "\nTry: \n   C:\\pip install pygame on a window command prompt.")

from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d

try:
    from BloomEffect.bloom import *
except ImportError:
    raise ImportError('\n<PyGameEffect> library is missing on your system.'
                      "\nTry: \n   C:\\pip install PyGameEffect on a window command prompt.")

import timeit
import os
import BloomEffect

PROJECT_PATH = list(BloomEffect.__path__)
os.chdir(PROJECT_PATH[0] + "\\tests")

width, height = 640, 400
screen = pygame.display.set_mode((width * 2, height))

pygame.display.set_caption("Testing blur algorithms")

background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
background = pygame.transform.smoothscale(background, (1280, 1024))
rgb_array = pixels3d(background)

print('\nTESTING WITH IMAGE 1280x1024')

N = 10

blur5x5_array24(rgb_array)

t = timeit.timeit("blur5x5_array24(rgb_array)",
                  "from __main__ import blur5x5_array24, rgb_array", number=N)
print("\nPerformance testing blur5x5_array24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

alpha_array = array_alpha(background)
rgb_array   = array3d(background)
rgba_array  = numpy.dstack((rgb_array, alpha_array)).transpose(1, 0, 2)
blur5x5_array32(rgb_array)
t = timeit.timeit("blur5x5_array32(rgb_array)",
                  "from __main__ import blur5x5_array32, rgb_array", number=N)
print("\nPerformance testing blur5x5_array32 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

alpha_array = pixels_alpha(background)
rgb_array   = pixels3d(background)
rgba_array  = numpy.dstack((rgb_array, alpha_array)).transpose(1, 0, 2)
blur5x5_array32_inplace(rgba_array)
t = timeit.timeit("blur5x5_array32_inplace(rgba_array)",
                  "from __main__ import blur5x5_array32_inplace, rgba_array", number=N)
print("\nPerformance testing blur5x5_array32_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

blur5x5_array24_inplace(rgba_array)
t = timeit.timeit("blur5x5_array24_inplace(rgba_array)",
                  "from __main__ import blur5x5_array24_inplace, rgba_array", number=N)
print("\nPerformance testing blur5x5_array24_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

# --------------- BLOOM EFFECT
bloom_effect24(background, 128)
t = timeit.timeit("bloom_effect24(background, 128)",
                  "from __main__ import bloom_effect24, background", number=N)
print("\nPerformance testing bloom_effect24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

# smooth = 10
t = timeit.timeit("bloom_effect24(background, 128, smooth_=10)",
                  "from __main__ import bloom_effect24, background", number=N)
print("\nPerformance testing bloom_effect24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

# Fast flag
t = timeit.timeit("bloom_effect24(background, 128, fast_=True)",
                  "from __main__ import bloom_effect24, background", number=N)
print("\nPerformance testing bloom_effect24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

t = timeit.timeit("bloom_effect24(background, 128)",
                  "from __main__ import bloom_effect24, background", number=N)
print("\nPerformance testing bloom_effect24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

# smooth = 10
bloom_effect32(background, 128, smooth_=10)
t = timeit.timeit("bloom_effect32(background, 128, smooth_=10)",
                  "from __main__ import bloom_effect32, background", number=N)
print("\nPerformance testing bloom_effect32 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

# Fast flag
t = timeit.timeit("bloom_effect32(background, 128, fast_=True)",
                  "from __main__ import bloom_effect32, background", number=N)
print("\nPerformance testing bloom_effect32 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
background = pygame.transform.smoothscale(background, (1280, 1024))
bloom_effect24_inplace(background, 128, fast_=False)
t = timeit.timeit("bloom_effect24_inplace(background, 128, fast_=False)",
                  "from __main__ import bloom_effect24_inplace, background", number=N)
print("\nPerformance testing bloom_effect24_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

t = timeit.timeit("bloom_effect24_inplace(background, 128, fast_=True)",
                  "from __main__ import bloom_effect24_inplace, background", number=N)
print("\nPerformance testing bloom_effect24_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

bloom_effect32_inplace(background, 128, fast_=False)
t = timeit.timeit("bloom_effect32_inplace(background, 128, fast_=False)",
                  "from __main__ import bloom_effect32_inplace, background", number=N)
print("\nPerformance testing bloom_effect32_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

t = timeit.timeit("bloom_effect32_inplace(background, 128, fast_=True)",
                  "from __main__ import bloom_effect32_inplace, background", number=N)
print("\nPerformance testing bloom_effect32_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

from BloomEffect.bloom import test_bpf24_c, test_bpf32_c, test_bpf24_inplace, \
    test_bpf32_inplace, filtering24, filtering32, test_array32_rescale

# ------------------ OTHER FUNCTIONS
test_bpf24_c(rgb_array, 128)
t = timeit.timeit("test_bpf24_c(rgb_array, 128)",
                  "from __main__ import test_bpf24_c, rgb_array", number=N)
print("\nPerformance testing test_bpf24_c per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

test_bpf32_c(background, 128)
t = timeit.timeit("test_bpf32_c(background, 128)",
                  "from __main__ import test_bpf32_c, background", number=N)
print("\nPerformance testing test_bpf32_c per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

test_bpf24_inplace(rgb_array, 128)
t = timeit.timeit("test_bpf24_inplace(rgb_array, 128)",
                  "from __main__ import test_bpf24_inplace, rgb_array", number=N)
print("\nPerformance testing test_bpf24_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

test_bpf32_inplace(background, 128)
t = timeit.timeit("test_bpf32_inplace(background, 128)",
                  "from __main__ import test_bpf32_inplace, background", number=N)
print("\nPerformance testing test_bpf32_inplace per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))


mask = build_mask_from_surface(background, False)
build_mask_from_surface(background, False)
t = timeit.timeit("build_mask_from_surface(background, False)",
                  "from __main__ import build_mask_from_surface, background", number=N)
print("\nPerformance testing build_mask_from_surface per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

filtering24(background, mask)
t = timeit.timeit("filtering24(background, mask)",
                  "from __main__ import filtering24, background, mask", number=N)
print("\nPerformance testing filtering24 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
filtering32(background, mask)
t = timeit.timeit("filtering32(background, mask)",
                  "from __main__ import filtering32, background, mask", number=N)
print("\nPerformance testing filtering32 per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))

background = pygame.image.load('../Assets/Aliens.jpg').convert_alpha()
background = pygame.transform.smoothscale(background, (800, 600))
rgb_array = pixels3d(background)
test_array32_rescale(rgb_array, 1280, 1024)
t = timeit.timeit("test_array32_rescale(rgb_array, 1280, 1024)",
                  "from __main__ import test_array32_rescale, rgb_array", number=N)
print("\nPerformance testing test_array32_rescale per call %s overall time %s for %s"
      % (round(float(t)/float(N), 10), round(float(t), 5), N))
pygame.quit()
