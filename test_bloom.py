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
    print("\n<pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")
    raise SystemExit

try:
    import numpy
    from numpy import asarray, uint8, float32, zeros, float64
except ImportError:
    print("\n<numpy> library is missing on yor system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")


import bloom


try:
    from bloom import bloom_effect_buffer24, bloom_effect_buffer32, bloom_effect_array24, \
            bloom_effect_array32, blur5x5_buffer24, scale_array24_mult
except ImportError as e:
    print("\n ", e)

import timeit
import os

x = 601
y = 600

os.environ['SDL_VIDEODRIVER'] = 'windib'
pygame.display.init()

im = pygame.image.load("i2.jpg")
im = pygame.transform.smoothscale(im, (x, y))
w, h = im.get_size()
screen = pygame.display.set_mode((w * 2, h), pygame.SWSURFACE, 32)
im = im.convert(24)


N = 100
print("tostring ", timeit.timeit("pygame.image.tostring(im, 'RGB')",
                                 "from __main__ import pygame, im", number=N) / N)
N = 100000
print("get_view('2') ", timeit.timeit("im.get_view('2')",
                                      "from __main__ import pygame, im", number=N) / N)
print("get_view('3') ", timeit.timeit("im.get_view('3')",
                                      "from __main__ import pygame, im", number=N) / N)

N = 10
#
# print("bloom_effect_buffer24 ",
#       timeit.timeit("bloom_effect_buffer24(im, 255, smooth_=1)",
#                     "from __main__ import bloom_effect_buffer24, im", number=N) / N)
#
# print("bloom_effect_buffer24 ",
#       timeit.timeit("bloom_effect_buffer24(im, 255, smooth_=10)",
#                     "from __main__ import bloom_effect_buffer24, im", number=N) / N)
#
# print("bloom_effect_array24 smooth = 1",
#       timeit.timeit("bloom_effect_array24(im, 255, smooth_=1)",
#                     "from __main__ import bloom_effect_array24, im", number=N) / N)
#
# print("bloom_effect_array24 smooth = 10",
#       timeit.timeit("bloom_effect_array24(im, 255, smooth_=10)",
#                     "from __main__ import bloom_effect_array24, im", number=N) / N)
# im = im.convert_alpha()
# print("bloom_effect_array32 smooth = 1",
#       timeit.timeit("bloom_effect_array32(im, 255, smooth_=1)",
#                     "from __main__ import bloom_effect_array32, im", number=N) / N)
#
# print("bloom_effect_array32 smooth = 10",
#       timeit.timeit("bloom_effect_array32(im, 255, smooth_=10)",
#                     "from __main__ import bloom_effect_array32, im", number=N) / N)

im = pygame.image.load("i2.jpg")
im = pygame.transform.smoothscale(im, (x, y)).convert(24)

CLOCK = pygame.time.Clock()
i = 0
j = 255
STOP_DEMO = True
while STOP_DEMO:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    bloom2_x10 = bloom_effect_array24(im, j, smooth_=1)  # smooth_=10
    screen.blit(bloom2_x10, (0, 0))

    pygame.display.flip()

    if keys[pygame.K_ESCAPE]:
        STOP_DEMO = False

    if keys[pygame.K_F8]:
        pygame.image.save(screen, 'Screendump' + str(i) + '.png')

    i += 1
    j -= 1

    if j < 0:
        j = 255

    CLOCK.tick()
    time_ = CLOCK.get_time()
    fps_ = CLOCK.get_fps()
    print(fps_)
