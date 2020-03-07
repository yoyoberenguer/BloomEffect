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
    import numpy
    from numpy import asarray, uint8, float32, zeros, float64
except ImportError:
    print("\n<numpy> library is missing on yor system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

try:
    import bloom
    from bloom import bloom_effect_buffer, bloom_effect_array
except ImportError:
    print("\nHave you build the project?"
          "\nC:>python setup_bloom.py build_ext --inplace")
import timeit

im = pygame.image.load("I2.jpg")
im = pygame.transform.smoothscale(im, (600, 600))

w, h = im.get_size()
screen = pygame.display.set_mode((w, h))

print(timeit.timeit("bloom_effect_array(im, 255, smooth_=1)",
                    "from __main__ import bloom_effect_array, im", number=10) / 10)

print(timeit.timeit("bloom_effect_buffer(im, 255, smooth_=1)",
                    "from __main__ import bloom_effect_buffer, im", number=10) / 10)

i = 0
j = 255
STOP_DEMO = True
while STOP_DEMO:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    # screen.fill((10, 10, 10, 255))
    org = bloom_effect_array(im, j, smooth_=1)
    screen.blit(org, (0, 0))
    pygame.display.flip()

    if keys[pygame.K_ESCAPE]:
        STOP_DEMO = False

    if keys[pygame.K_F8]:
        pygame.image.save(screen, 'Screendump' + str(i) + '.png')

    i += 1
    j -= 1

    if j < 0:
        j = 255
